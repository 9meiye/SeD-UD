import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.base import DataManager
import importlib
import sys
from data.BERTencoder import TextEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from data.__init__ import benchmarks
from losses.total_loss import compute_loss
import torch.nn.functional as F
import random
import os

def load_config(dataset_name="MIntRec"):
    config_module_name = f"configs-{dataset_name}"
    config_module = importlib.import_module(config_module_name)
    return config_module.args

args = load_config()

def set_random_seed(seed=42):
    print(f"Setting random seed: {seed}")
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set: {seed}")

try:
    from utils.Function import EarlyStopping
except ImportError:
    class EarlyStopping:
        def __init__(self, patience=3, delta=0.001):
            self.patience = patience
            self.delta = delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False
        def __call__(self, val_metric):
            if self.best_score is None:
                self.best_score = val_metric
            elif val_metric < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_metric
                self.counter = 0

try:
    from utils.alignment import AlignSubNet
except ImportError:
    class AlignSubNet:
        def __init__(self, mode=None, text_feat_dim=None, video_feat_dim=None, audio_feat_dim=None, dst_len=None):
            pass
        def forward(self, *args, **kwargs):
            
            return None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenoisingBottleneck(nn.Module):
    def __init__(self, input_dim, min_bottleneck=64, max_bottleneck=768, tau=1.0, eps=1e-6):
        super().__init__()
        self.min_bottleneck = min_bottleneck
        self.max_bottleneck = max_bottleneck
        self.tau = tau
        self.eps = eps
        
        self.W_p = nn.Linear(input_dim, input_dim)
        self.b_p = nn.Parameter(torch.zeros(input_dim))
        
        self.W_r = nn.Linear(input_dim * 2, 1)
        self.b_r = nn.Parameter(torch.zeros(1))
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_bottleneck),
            nn.Softmax(dim=-1)
        )
        
        self.encoder = nn.Linear(input_dim, max_bottleneck)
        self.decoder = nn.Linear(max_bottleneck, input_dim)
        self.actual_bottleneck_dim = max_bottleneck
        
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.b2 = nn.Parameter(torch.tensor(0.0))
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        self.external_noise_level = None
        self.external_correlation_score = None
        
        self.bottleneck_dims = []
        
        self.train_attention_times = []   
        self.train_sorting_times = []     
        self.eval_attention_times = []    
        self.eval_sorting_times = []      
        
        self._current_loss = None
        self._importance_weights = None
        self._parameter_ranking = None
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        if self._importance_weights is not None:
            state_dict[prefix + '_importance_weights'] = self._importance_weights
        if self._parameter_ranking is not None:
            state_dict[prefix + '_parameter_ranking'] = self._parameter_ranking
        
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        importance_key = prefix = ''
        for key in state_dict.keys():
            if key.endswith('_importance_weights'):
                importance_key = key
                break
        
        ranking_key = ''
        for key in state_dict.keys():
            if key.endswith('_parameter_ranking'):
                ranking_key = key
                break
        
        importance_weights = None
        parameter_ranking = None
        
        if importance_key:
            importance_weights = state_dict.pop(importance_key)
        
        if ranking_key:
            parameter_ranking = state_dict.pop(ranking_key)
        
        result = super().load_state_dict(state_dict, strict=strict)
        
        if importance_weights is not None:
            self._importance_weights = importance_weights
        
        if parameter_ranking is not None:
            self._parameter_ranking = parameter_ranking
        
        return result
    def tempered_normalization(self, alpha):
        tanh_alpha = torch.tanh(alpha / self.tau)
        
        norm = torch.norm(tanh_alpha, p=2)
        
        alpha_bar = tanh_alpha / (norm + self.eps)
        return alpha_bar
    def compute_beta(self, alpha_bar):
        beta = self.w2 * F.silu(self.w1 * alpha_bar + self.b1) + self.b2
        return beta
    def compute_compression_dimension(self, beta):
        D = self.max_bottleneck
        
        D_pow = D ** (1 - beta)
        
        D_rounded = torch.round(D_pow)
        
        Dc = torch.clamp(D_rounded, self.min_bottleneck, self.max_bottleneck)
        return Dc.int()
    def compute_parameter_importance(self, loss):
        self._current_loss = loss
        
        loss.backward(retain_graph=True)
        
        if self.encoder.weight.grad is not None:
            
            enc_dim_importance = (self.encoder.weight.norm(p=2, dim=1)**2 * 
                                self.encoder.weight.grad.norm(p=2, dim=1)**2)
        else:
            enc_dim_importance = torch.zeros(self.max_bottleneck, device=self.encoder.weight.device)
        
        if self.decoder.weight.grad is not None:
            
            dec_dim_importance = (self.decoder.weight.norm(p=2, dim=0)**2 *
                                self.decoder.weight.grad.norm(p=2, dim=0)**2)
        else:
            dec_dim_importance = torch.zeros(self.max_bottleneck, device=self.decoder.weight.device)
        
        combined_importance = enc_dim_importance + dec_dim_importance
        
        if combined_importance.sum() > 0:
            normalized_importance = combined_importance / combined_importance.sum()
        else:
            normalized_importance = torch.ones(self.max_bottleneck, device=combined_importance.device) / self.max_bottleneck
        
        self._importance_weights = normalized_importance.detach()
        
        self.zero_grad()
        return normalized_importance
    
    def compute_noise_intensity(self, x):
        I = torch.sigmoid(self.W_p(x) + self.b_p)
        
        D = x.shape[-1]
        weighted_abs = I * torch.abs(x)
        gamma = torch.sigmoid(torch.sum(weighted_abs, dim=-1) / D)
        
        return gamma, I
    
    def _single_modal_denoise(self, x, original_dim):
        if x.dim() == 3:
            x = x.mean(dim=1)  
        
        if x.shape[-1] != original_dim:
            x = F.pad(x, (0, original_dim - x.shape[-1]))
        
        if torch.isnan(x).any():
            print(f"Warning: Single modal denoising input contains NaN values")
            x = torch.nan_to_num(x, nan=0.0)
        
        gamma, I = self.compute_noise_intensity(x)
        
        if torch.isnan(gamma).any():
            print(f"Warning: Noise intensity contains NaN values, using default values")
            gamma = torch.ones_like(gamma) * 0.5
        
        alpha_bar = self.tempered_normalization(gamma.mean())
        
        beta = self.compute_beta(alpha_bar)
        
        bottleneck_dim = self.compute_compression_dimension(beta)
        
        if bottleneck_dim.dim() > 0:
            bottleneck_dim = bottleneck_dim.item()
        
        bottleneck_dim = min(bottleneck_dim, self.max_bottleneck)
        
        if bottleneck_dim < self.max_bottleneck:
            
            if self._importance_weights is not None:
                
                importance_weights = self._importance_weights
            else:
                
                
                importance_weights = torch.ones(self.max_bottleneck, device=x.device) / self.max_bottleneck
            
            _, topk_indices = torch.topk(importance_weights, k=bottleneck_dim)
            
            enc_weight = self.encoder.weight[topk_indices]
            enc_bias = self.encoder.bias[topk_indices]
            dec_weight = self.decoder.weight[:, topk_indices]
            dec_bias = self.decoder.bias
        else:
            
            enc_weight = self.encoder.weight
            enc_bias = self.encoder.bias
            dec_weight = self.decoder.weight
            dec_bias = self.decoder.bias
        
        encoded = F.linear(x, weight=enc_weight, bias=enc_bias)
        encoded = self.dropout(self.act(encoded))
        decoded = F.linear(encoded, weight=dec_weight, bias=dec_bias)
        decoded = self.act(decoded)
        if self.training:  
            decoded = self.dropout(decoded)
        
        compression_ratio = 1 - (bottleneck_dim / self.max_bottleneck)
        if self.modal_type == 'Text' and self.text_ratios is not None:
            self.text_ratios.append(compression_ratio)
        elif self.modal_type == 'Video' and self.video_ratios is not None:
            self.video_ratios.append(compression_ratio)
        elif self.modal_type == 'Audio' and self.audio_ratios is not None:
            self.audio_ratios.append(compression_ratio)
        
        self._last_noise_level = gamma.mean().item()
        return decoded
    
    def compute_redundancy_degree(self, F_pri, F_aux1, F_aux2):
        D = F_pri.shape[-1]
        
        attn_weights1 = torch.matmul(F_pri, F_aux1.transpose(-2, -1)) / (D ** 0.5)
        attn_probs1 = F.softmax(attn_weights1, dim=-1)
        V1 = torch.matmul(attn_probs1, F_aux1)
        
        attn_weights2 = torch.matmul(F_pri, F_aux2.transpose(-2, -1)) / (D ** 0.5)
        attn_probs2 = F.softmax(attn_weights2, dim=-1)
        V2 = torch.matmul(attn_probs2, F_aux2)
        
        V_concat = torch.cat([V1, V2], dim=-1)
        r = torch.sigmoid(self.W_r(V_concat) + self.b_r)
        
        return r.squeeze(-1), V1, V2
    
    def _multi_modal_redundancy_removal(self, x, y, z, modal_type):
        if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(z).any():
            print(f"Warning: Redundancy removal input contains NaN values")
            x = torch.nan_to_num(x, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            z = torch.nan_to_num(z, nan=0.0)
        
        r, V1, V2 = self.compute_redundancy_degree(x, y, z)
        
        if torch.isnan(r).any():
            print(f"Warning: Redundancy degree contains NaN values, using default values")
            r = torch.ones_like(r) * 0.5
        
        alpha_bar = self.tempered_normalization(r.mean())
        
        beta = self.compute_beta(alpha_bar)
        
        bottleneck_dim = self.compute_compression_dimension(beta)
        
        if bottleneck_dim.dim() > 0:
            bottleneck_dim = bottleneck_dim.item()
        
        if self.training:
            self.bottleneck_dims.append(bottleneck_dim)
        
        if bottleneck_dim < self.max_bottleneck:
            
            if self._importance_weights is not None:
                
                importance_weights = self._importance_weights
            else:
                
                
                importance_weights = torch.ones(self.max_bottleneck, device=x.device) / self.max_bottleneck
            
            _, topk_indices = torch.topk(importance_weights, k=bottleneck_dim)
            
            enc_weight = self.encoder.weight[topk_indices]
            enc_bias = self.encoder.bias[topk_indices]
            dec_weight = self.decoder.weight[:, topk_indices]
            dec_bias = self.decoder.bias
        else:
            
            enc_weight = self.encoder.weight
            enc_bias = self.encoder.bias
            dec_weight = self.decoder.weight
            dec_bias = self.decoder.bias
        
        encoded = F.linear(x, weight=enc_weight, bias=enc_bias)
        encoded = self.dropout(self.act(encoded))
        
        decoded = F.linear(encoded, weight=dec_weight, bias=dec_bias)
        decoded = self.act(decoded)
        if self.training:  
            decoded = self.dropout(decoded)
        
        compression_ratio = 1 - (bottleneck_dim / self.max_bottleneck)
        if self.modal_type == 'Text' and self.text_ratios is not None:
            self.text_ratios.append(compression_ratio)
        elif self.modal_type == 'Video' and self.video_ratios is not None:
            self.video_ratios.append(compression_ratio)
        elif self.modal_type == 'Audio' and self.audio_ratios is not None:
            self.audio_ratios.append(compression_ratio)
        
        self._last_correlation_scores = r.mean().item()
        return decoded
    def forward(self, x, y, z, modal_type):
        self.modal_type = modal_type  
        
        self.text_ratios = [] if modal_type == 'Text' else None
        self.video_ratios = [] if modal_type == 'Video' else None 
        self.audio_ratios = [] if modal_type == 'Audio' else None
        
        original_dim = x.shape[-1]
        if y is None and z is None:
            
            decoded = self._single_modal_denoise(x, original_dim)
        else:
            
            decoded = self._multi_modal_redundancy_removal(x, y, z, modal_type)
        return decoded
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.MAG = MAG(args)
        self.text_compressor = DenoisingBottleneck(input_dim=768)
        self.video_compressor = DenoisingBottleneck(input_dim=768)
        self.audio_compressor = DenoisingBottleneck(input_dim=768)
        self.text_compressor.modal_type = 'Text'
        self.video_compressor.modal_type = 'Video' 
        self.audio_compressor.modal_type = 'Audio'
        self.denoising_bottleneck = DenoisingBottleneck(input_dim=768)
        self.fusion_norm = nn.LayerNorm(768)
        self.text_fusion_norm = nn.LayerNorm(768)
        self.text_encoder = TextEncoder(pretrained_model=args.text_pretrained_model)
        
        self.align_net = AlignSubNet(
            mode='ctc',
            text_feat_dim=768,
            video_feat_dim=768,
            audio_feat_dim=768,
            dst_len=512  
        )
        
        self.video_adapter = nn.Linear(args.video_feat_dim, 768)
        self.video_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 768))
        
        self.audio_adapter = nn.Linear(args.audio_feat_dim, 768)
        self.audio_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 768))
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes))
        self.denoised_classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes))
        self.text_classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes))
        
        self.concat_projection = nn.Linear(768*3, 768)
    def fused(self, x, y, z):
        return self.MAG(x, y, z)
    def forward(self, text_input_ids, text_attention_mask, video_feats, audio_feats, noise_level=0.0):
        
        output_dict = {
            'fused_feats': None,
            'denoised_feats': None,
            'text_feats': None,
            'fused_preds': None,
            'denoised_preds': None,
            'text_preds': None,
            'video_feats': None,
            'audio_feats': None,
            'denoised_text_feats': None,
            'denoised_video_feats': None,
            'denoised_audio_feats': None,
            'text_ratios': [],
            'video_ratios': [],
            'audio_ratios': []
        }
        
        
        text_embedding = self.text_encoder.get_embeddings(text_input_ids.long())
        
        video = self.video_adapter(video_feats.float())
        video = self.video_encoder(video).squeeze(2)
        
        audio = self.audio_adapter(audio_feats.float())
        audio = self.audio_encoder(audio)
        
        text = self.text_encoder(
            input_ids=text_input_ids.long(),
            attention_mask=text_attention_mask.long(),
            inputs_embeds=None
        )
        
        if isinstance(text, tuple):
            text = text[0]  
        if noise_level > 0:
            
            std_dev = torch.sqrt(torch.tensor(noise_level, device=text.device))
            text = text + torch.randn_like(text) * std_dev
        
        video = video.mean(dim=1)
        audio = audio.mean(dim=1)
        
        output_dict.update({
            'text_feats': self.text_fusion_norm(text),
            'video_feats': video,
            'audio_feats': audio,
            'text_preds': self.text_classifier(text)
        })
        
        deredundant_text = self.text_compressor(text, video, audio, 'Text')
        deredundant_video = self.video_compressor(video, text, audio, 'Video')
        deredundant_audio = self.audio_compressor(audio, text, video, 'Audio')
        fused = self.fused(deredundant_text, deredundant_video, deredundant_audio)
        output_dict.update({
            'denoised_text_feats': deredundant_text,
            'denoised_video_feats': deredundant_video,
            'denoised_audio_feats': deredundant_audio,
            'text_ratios': self.text_compressor.text_ratios,
            'video_ratios': self.video_compressor.video_ratios,
            'audio_ratios': self.audio_compressor.audio_ratios
        })
        
        if hasattr(self.text_compressor, '_last_correlation_scores'):
            output_dict['correlation_scores'] = self.text_compressor._last_correlation_scores
        
        fused = self.fusion_norm(fused)
        output_dict.update({
            'fused_feats': fused,
            'fused_preds': self.classifier(fused)
        })
        
        denoised_feats = self.denoising_bottleneck(fused, None, None, 'Fused')
        output_dict['denoised_feats'] = denoised_feats
        output_dict['denoised_preds'] = self.denoised_classifier(denoised_feats)
        
        if hasattr(self.denoising_bottleneck, '_last_noise_level'):
            output_dict['noise_level'] = self.denoising_bottleneck._last_noise_level
        
        if hasattr(self.text_compressor, 'text_ratios'):
            self.text_compressor.text_ratios.clear()
        if hasattr(self.video_compressor, 'video_ratios'):
            self.video_compressor.video_ratios.clear()
        if hasattr(self.audio_compressor, 'audio_ratios'):
            self.audio_compressor.audio_ratios.clear()
        return output_dict
class MAG(nn.Module):
    def __init__(self, args):
        super(MAG, self).__init__()
        self.args = args
        
        text_feat_dim, audio_feat_dim, video_feat_dim = 768, 768, 768
        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)
        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)
        
        self.beta_shift = 0.006
        
        self.residual = nn.Linear(768, 768)
        self.LayerNorm = nn.LayerNorm(768) 
        self.dropout = nn.Dropout(args.hidden_dropout_prob) 
        
        self.align_net = AlignSubNet(
            mode='ctc',
            text_feat_dim=768,
            video_feat_dim=768,
            audio_feat_dim=768,
            dst_len=args.max_cons_seq_length
        )
    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        if text_embedding.size(0)<128:
            hm_norm = hm_norm[:text_embedding.size(0)]
        
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)
        
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)
        acoustic_vis_embedding = alpha * h_m
        
        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + self.residual(text_embedding))
        )
        return embedding_output
def collate_fn(batch):
    text_feats = [item['text_feats'] for item in batch]
    return {
        'input_ids': torch.stack([t[0] for t in text_feats]),
        'attention_mask': torch.stack([t[1] for t in text_feats]),
        'video_feats': torch.stack([item['video_feats'] for item in batch]),
        'audio_feats': torch.stack([item['audio_feats'] for item in batch]),
        'label': torch.tensor([item['label_ids'] for item in batch])
    }
def compute_accuracy(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'text_input_ids': batch['input_ids'].to(device),
                'text_attention_mask': batch['attention_mask'].to(device),
                'video_feats': batch['video_feats'].to(device),
                'audio_feats': batch['audio_feats'].to(device)
            }
            
            outputs = model(**inputs)
            
            logits = outputs['fused_preds']
            
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).cpu().numpy()
            label = batch['label'].cpu().numpy()
            preds.extend(pred)
            labels.extend(label)
    preds = np.array(preds)
    labels = np.array(labels)
    
    accuracy = accuracy_score(labels, preds)
    precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
from torch.utils.tensorboard import SummaryWriter
def compute_parameter_importance_converged(model, data_loader, device, num_batches=3):
    model.eval()
    
    bottlenecks = [
        model.text_compressor,
        model.video_compressor,
        model.audio_compressor,
        model.denoising_bottleneck
    ]
    
    importance_accumulators = {}
    for i, bottleneck in enumerate(bottlenecks):
        importance_accumulators[i] = {
            'enc_weight': torch.zeros(bottleneck.max_bottleneck, device=device),
            'enc_bias': torch.zeros(bottleneck.max_bottleneck, device=device),
            'dec_weight': torch.zeros(bottleneck.max_bottleneck, device=device)
        }
    
    valid_batches = 0
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        try:
            inputs = {
                'text_input_ids': batch['input_ids'].to(device),
                'text_attention_mask': batch['attention_mask'].to(device),
                'video_feats': batch['video_feats'].to(device),
                'audio_feats': batch['audio_feats'].to(device)
            }
            labels = batch['label'].to(device)
            
            model.zero_grad()
            outputs = model(**inputs)
            loss, _ = compute_loss(outputs, labels, denoised=True, deredundant=True)
            
            loss.backward()
            
            for i, bottleneck in enumerate(bottlenecks):
                if bottleneck.encoder.weight.grad is not None:
                    enc_weight_importance = (bottleneck.encoder.weight.norm(p=2, dim=1)**2 * 
                                             bottleneck.encoder.weight.grad.norm(p=2, dim=1)**2)
                    importance_accumulators[i]['enc_weight'] += enc_weight_importance.detach()
                    
                    if bottleneck.encoder.bias is not None and bottleneck.encoder.bias.grad is not None:
                        enc_bias_importance = (bottleneck.encoder.bias.norm(p=2)**2 * 
                                               bottleneck.encoder.bias.grad.norm(p=2)**2)
                        enc_bias_expanded = enc_bias_importance.repeat(bottleneck.max_bottleneck)
                        importance_accumulators[i]['enc_bias'] += enc_bias_expanded.detach()
                    
                    dec_weight_importance = (bottleneck.decoder.weight.norm(p=2, dim=0)**2 *
                                             bottleneck.decoder.weight.grad.norm(p=2, dim=0)**2)
                    importance_accumulators[i]['dec_weight'] += dec_weight_importance.detach()
            
            valid_batches += 1
            model.zero_grad()
            
        except Exception as e:
            print(f"Warning: Error processing batch {batch_idx}: {e}")
            continue
    
    if valid_batches == 0:
        print("Warning: No valid batches for importance computation")
        return 0
    
    computed_count = 0
    for i, bottleneck in enumerate(bottlenecks):
        if valid_batches > 0:
            avg_enc_weight = importance_accumulators[i]['enc_weight'] / valid_batches
            avg_enc_bias = importance_accumulators[i]['enc_bias'] / valid_batches
            avg_dec_weight = importance_accumulators[i]['dec_weight'] / valid_batches
            
            combined_importance = avg_enc_weight + avg_enc_bias + avg_dec_weight
            
            if combined_importance.sum() > 0:
                normalized_importance = combined_importance / combined_importance.sum()
            else:
                normalized_importance = torch.ones(bottleneck.max_bottleneck, 
                                                   device=combined_importance.device) / bottleneck.max_bottleneck
            
            bottleneck._importance_weights = normalized_importance.detach()
            bottleneck._parameter_ranking = torch.argsort(normalized_importance, descending=True)
            
            computed_count += 1
    
    model.zero_grad()
    return computed_count

def train_and_validate(data_manager):
    
    current_seed = getattr(args, 'seed', 42)
    print(f"=== Training Started ===")
    print(f"Current random seed: {current_seed}")
    print(f"Using full SeD-UD model (both denoising and redundancy removal)")
    
    writer = SummaryWriter(log_dir=f'runs/experiment_sed_ud')
    
    train_loader = DataLoader(data_manager.mm_data['train'], 
                            batch_size=args.train_batch_size, 
                            shuffle=True, 
                            collate_fn=collate_fn)
    val_loader = DataLoader(data_manager.mm_data['dev'], 
                          batch_size=args.eval_batch_size, 
                          collate_fn=collate_fn)
    
    model = MultiModalClassifier(num_classes=args.num_labels).to(device)
    
    experiment_config = {
        'seed': current_seed
    }
   
    optimizer = optim.AdamW(model.parameters(), 
                          lr=1e-5, 
                          weight_decay=0.2,  
                          eps=1e-9)  
    
    def lr_lambda(epoch):
        if epoch < 5:  
            return 0.01 + (1 - 0.01) * epoch / 5
        else:  
            progress = (epoch - 5) / (args.num_train_epochs - 5)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=10, delta=0.005)
    
    for epoch in range(args.num_train_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in train_loader:
            inputs = {
                'text_input_ids': batch['input_ids'].to(device),
                'text_attention_mask': batch['attention_mask'].to(device),
                'video_feats': batch['video_feats'].to(device),
                'audio_feats': batch['audio_feats'].to(device)
            }
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss, _ = compute_loss(outputs, labels, denoised=True, deredundant=True)  
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  
            train_loss += loss.item()
            
            preds_key = 'fused_preds'  
            
            if preds_key not in outputs:
                raise KeyError(f"Prediction key {preds_key} does not exist in model output")
            train_correct += (outputs[preds_key].argmax(1) == labels).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        
        model.eval()
        val_metrics = compute_accuracy(model, val_loader)
        
        scheduler.step()
        
        early_stopping(val_metrics['accuracy'])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            import os
            os.makedirs('./model', exist_ok=True)
            
            computed = compute_parameter_importance_converged(model, train_loader, device)
            if computed > 0:
                print(f"Training completed: Computed parameter importance for {computed} bottlenecks")
            
            torch.save(model.state_dict(), './model/best_model.pth')
            print(f"Final model saved with parameter importance weights")
            break
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2%} | "
              f"Val Acc: {val_metrics['accuracy']:.2%} | "
              f"Val F1: {val_metrics['f1_weighted']:.4f}")
def test_model(data_manager):
    
    test_loader = DataLoader(data_manager.mm_data['test'], batch_size=args.test_batch_size, collate_fn=collate_fn)
    
    model = MultiModalClassifier(num_classes=args.num_labels).to(device)
    
    import os
    model_path = './model/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    
    print(f"Model loaded from: {model_path} (with parameter importance weights)")
    
    print("\n=== Model Loading Check ===")
    print(f"Model parameters loaded successfully, ready to test with trained model")
    
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'text_input_ids': batch['input_ids'].to(device),
                'text_attention_mask': batch['attention_mask'].to(device),
                'video_feats': batch['video_feats'].to(device),
                'audio_feats': batch['audio_feats'].to(device)
            }
            labels = batch['label'].cpu().numpy()
            
            outputs = model(**inputs)
            
            preds_key = 'fused_preds'
            
            if preds_key not in outputs:
                raise KeyError(f"Prediction key {preds_key} does not exist in model output")
            
            probs = torch.softmax(outputs[preds_key], dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
    
    label_list = benchmarks[args.dataset]['labels']
    
    print("\n=== Test Results ===")
    print(classification_report(all_labels, all_preds, target_names=label_list, digits=4))
    
    test_metrics = compute_accuracy(model, test_loader)
    
    print("\n=== Comprehensive Evaluation Metrics (using sklearn functions) ===")
    print(f"Accuracy (ACC): {test_metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {test_metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {test_metrics['recall_weighted']:.4f}")
    print(f"Weighted F1 Score: {test_metrics['f1_weighted']:.4f}")
def main():
    seed = getattr(args, 'seed', 42)  
    set_random_seed(seed)
    
    print("=== Dataset Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    data = DataManager(args)
    print(f"Dataset loaded:\n"
          f"Training set: {len(data.mm_data['train'])} samples\n"
          f"Validation set: {len(data.mm_data['dev'])} samples\n"
          f"Test set: {len(data.mm_data['test'])} samples")
    
    train_and_validate(data)
    
    test_model(data)

if __name__ == '__main__':
    main()
