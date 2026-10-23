import importlib
import math
import os
import random
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from data.__init__ import benchmarks
from data.BERTencoder import TextEncoder
from data.base import DataManager
from losses.total_loss import compute_loss
from utils.Function import EarlyStopping


def load_config(dataset_name="MIntRec"):
    config_module = importlib.import_module(f"configs-{dataset_name}")
    return config_module.args


args = load_config(os.environ.get("SEDUD_DATASET", "MIntRec"))
device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
DIM = 768
DENOISED_KEY = "denoised_preds"


def to_device(batch):
    return {
        "text_input_ids": batch["input_ids"].to(device),
        "text_attention_mask": batch["attention_mask"].to(device),
        "video_feats": batch["video_feats"].to(device),
        "audio_feats": batch["audio_feats"].to(device),
    }


def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def make_bottleneck(config):
    return DenoisingBottleneck(
        DIM,
        d_min=config.idab_min_dim,
        d_max=config.idab_max_dim,
        b2_init=config.idab_b2_init,
    )

class RedundancyDegree(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.W_r = nn.Linear(2 * dim, 1)

    def forward(self, primary, auxiliary_1, auxiliary_2):
        dim = primary.shape[-1]
        attention_1 = torch.softmax(
            primary @ auxiliary_1.transpose(-2, -1) / math.sqrt(dim), dim=-1
        )
        value_1 = attention_1 @ auxiliary_1
        attention_2 = torch.softmax(
            primary @ auxiliary_2.transpose(-2, -1) / math.sqrt(dim), dim=-1
        )
        value_2 = attention_2 @ auxiliary_2
        return torch.sigmoid(
            self.W_r(torch.cat([value_1, value_2], dim=-1))
        ).squeeze(-1)


class NoiseIntensity(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.W_p = nn.Linear(dim, dim)

    def forward(self, features):
        importance = torch.sigmoid(self.W_p(features))
        return torch.sigmoid(
            (importance * features.abs()).sum(dim=-1) / features.shape[-1]
        )


class DenoisingBottleneck(nn.Module):
    def __init__(
        self,
        dim=768,
        d_min=64,
        d_max=768,
        tau=1.0,
        eps=1e-6,
        mask_temp=0.5,
        b2_init=-0.55,
    ):
        super().__init__()
        self.dim = int(dim)
        self.d_min = int(d_min)
        self.d_max = int(d_max if d_max is not None else dim)
        self.tau = float(tau)
        self.eps = float(eps)
        self.mask_temp = float(mask_temp)

        self.encoder = nn.Linear(self.dim, self.d_max)
        self.decoder = nn.Linear(self.d_max, self.dim)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.b2 = nn.Parameter(torch.tensor(float(b2_init)))

        self.register_buffer("_parameter_ranking", torch.arange(self.d_max))
        self.register_buffer("rank_ready", torch.zeros(1))

        self.selection_enabled = False

    def tempered_normalization(self, alpha):
        alpha = alpha.reshape(-1).float()
        values = torch.tanh(alpha / self.tau)
        norm = torch.linalg.vector_norm(values, ord=2)
        return values / (norm + self.eps)

    def compute_beta(self, alpha_bar):
        return self.w2 * F.silu(self.w1 * alpha_bar + self.b1) + self.b2

    def compute_compression_dimension(self, beta):
        return torch.clamp(self.dim ** (1.0 - beta), self.d_min, self.d_max)

    def forward(self, features, alpha):
        if features.dim() == 3:
            features = features.mean(dim=1)

        batch_size = features.shape[0]
        if alpha is None:
            alpha = torch.zeros(
                batch_size, device=features.device, dtype=features.dtype
            )
        alpha = alpha.reshape(-1)
        if alpha.numel() == 1:
            alpha = alpha.expand(batch_size)
        if alpha.numel() != batch_size:
            alpha = alpha[:batch_size]

        alpha_bar = self.tempered_normalization(alpha)
        beta = self.compute_beta(alpha_bar)
        soft_dim = self.compute_compression_dimension(beta)
        hard_dim = soft_dim.round().clamp(self.d_min, self.d_max)

        if self.selection_enabled:
            straight_through_dim = soft_dim + (hard_dim - soft_dim).detach()
            positions = torch.arange(
                self.d_max, device=features.device, dtype=features.dtype
            )
            soft_mask = torch.sigmoid(
                (straight_through_dim.unsqueeze(1) - positions.unsqueeze(0))
                / self.mask_temp
            )
            hard_mask = (
                positions.unsqueeze(0) < hard_dim.unsqueeze(1)
            ).to(features.dtype)
            mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            hard_dim = torch.full(
                (batch_size,), float(self.d_max), device=features.device
            )
            mask = torch.ones(
                batch_size,
                self.d_max,
                device=features.device,
                dtype=features.dtype,
            )

        permutation = self._parameter_ranking
        encoder_weight = self.encoder.weight[permutation]
        encoder_bias = self.encoder.bias[permutation]
        encoded = F.linear(features, encoder_weight, encoder_bias) * mask
        encoded = F.relu(encoded)

        decoder_weight = self.decoder.weight[:, permutation]
        decoded = F.relu(
            F.linear(encoded, decoder_weight, self.decoder.bias)
        )

        return decoded

class ModalityModulationFusion(nn.Module):
    def __init__(self, dim=768, heads=8, dropout=0.1, use_attention=True):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.gate_v = nn.Linear(2 * dim, 1)
        self.gate_a = nn.Linear(2 * dim, 1)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_a = nn.Linear(dim, dim)
        self.b_nt = nn.Parameter(torch.zeros(dim))

        self.mha = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, text, video, audio):
        gate_v = torch.sigmoid(
            self.gate_v(torch.cat([text, video], dim=-1))
        )
        gate_a = torch.sigmoid(
            self.gate_a(torch.cat([text, audio], dim=-1))
        )
        gated_video = gate_v * self.proj_v(video)
        gated_audio = gate_a * self.proj_a(audio)
        non_text = gated_video + gated_audio + self.b_nt

        if not self.use_attention:
            return self.norm1(text + non_text)

        query = text.unsqueeze(1)
        key_value = non_text.unsqueeze(1)
        attention, _ = self.mha(
            query, key_value, key_value, need_weights=False
        )
        hidden = self.norm1(
            text + self.drop(attention.squeeze(1))
        )
        return self.norm2(hidden + self.drop(self.ffn(hidden)))


class MultiModalClassifier(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        dim = DIM

        self.text_encoder = TextEncoder(
            pretrained_model=args.text_pretrained_model
        )
        self.video_adapter = nn.Linear(int(args.video_feat_dim), dim)
        self.audio_adapter = nn.Linear(int(args.audio_feat_dim), dim)
        dropout = float(args.hidden_dropout_prob)
        self.video_encoder = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, dim),
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, dim),
        )

        self.text_norm = nn.LayerNorm(dim)
        self.video_norm = nn.LayerNorm(dim)
        self.audio_norm = nn.LayerNorm(dim)
        self.rdc_text = RedundancyDegree(dim)
        self.rdc_video = RedundancyDegree(dim)
        self.rdc_audio = RedundancyDegree(dim)

        self.text_compressor = make_bottleneck(args)
        self.video_compressor = make_bottleneck(args)
        self.audio_compressor = make_bottleneck(args)
        self.denoising_bottleneck = make_bottleneck(args)

        self.nic = NoiseIntensity(dim)
        self.fusion = ModalityModulationFusion(
            dim,
            heads=args.fusion_heads,
            dropout=dropout,
            use_attention=args.use_cross_attention,
        )
        self.fusion_norm = nn.LayerNorm(dim)

        if args.classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(dim, args.classifier_hidden),
                nn.GELU(),
                nn.LayerNorm(args.classifier_hidden),
                nn.Dropout(dropout),
                nn.Linear(args.classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, num_classes)
            )
        self.use_pooler = bool(args.use_pooler)

    def bottlenecks(self):
        return {
            "text": self.text_compressor,
            "video": self.video_compressor,
            "audio": self.audio_compressor,
            "fused": self.denoising_bottleneck,
        }

    def set_selection(self, enabled):
        for bottleneck in self.bottlenecks().values():
            bottleneck.selection_enabled = bool(enabled)

    def enable_selection_if_ready(self):
        ready = all(
            float(bottleneck.rank_ready.item()) > 0
            for bottleneck in self.bottlenecks().values()
        )
        self.set_selection(ready)
        return ready

    def _encode_sequence(self, features, adapter, encoder, chunk=32):
        batch_size, time_steps, feature_dim = features.shape
        outputs = []
        for start in range(0, batch_size, chunk):
            part = features[start : start + chunk].reshape(-1, feature_dim)
            encoded = encoder(adapter(part))
            outputs.append(encoded.reshape(-1, time_steps, DIM).mean(dim=1))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        video_feats,
        audio_feats,
        noise_level=0.0,
    ):
        text = self.text_encoder(
            input_ids=text_input_ids.long(),
            attention_mask=text_attention_mask.long(),
        )
        if isinstance(text, tuple):
            text = text[0]
        if self.use_pooler:
            text = self.text_encoder.bert.pooler(text.unsqueeze(1))
        if noise_level > 0:
            text = text + torch.randn_like(text) * math.sqrt(noise_level)

        video = self._encode_sequence(
            video_feats.float(), self.video_adapter, self.video_encoder
        )
        audio = self._encode_sequence(
            audio_feats.float(), self.audio_adapter, self.audio_encoder
        )
        text = self.text_norm(text)
        video = self.video_norm(video)
        audio = self.audio_norm(audio)

        redundancy_text = self.rdc_text(text, video, audio)
        redundancy_video = self.rdc_video(video, text, audio)
        redundancy_audio = self.rdc_audio(audio, text, video)

        denoised_text = self.text_compressor(text, redundancy_text)
        denoised_video = self.video_compressor(video, redundancy_video)
        denoised_audio = self.audio_compressor(audio, redundancy_audio)

        fused = self.fusion_norm(
            self.fusion(denoised_text, denoised_video, denoised_audio)
        )
        noise_intensity = self.nic(fused)
        denoised = self.denoising_bottleneck(fused, noise_intensity)
        classify = self.classifier

        return {
            "text_feats": text,
            "video_feats": video,
            "audio_feats": audio,
            "denoised_text_feats": denoised_text,
            "denoised_video_feats": denoised_video,
            "denoised_audio_feats": denoised_audio,
            "fused_feats": fused,
            "denoised_feats": denoised,
            "text_preds": classify(text),
            "video_preds": classify(video),
            "audio_preds": classify(audio),
            "denoised_text_preds": classify(denoised_text),
            "denoised_video_preds": classify(denoised_video),
            "denoised_audio_preds": classify(denoised_audio),
            "fused_preds": classify(fused),
            DENOISED_KEY: classify(denoised),
            "redundancy_t": redundancy_text.detach(),
            "redundancy_v": redundancy_video.detach(),
            "redundancy_a": redundancy_audio.detach(),
            "noise_level": noise_intensity.detach(),
        }

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
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set: {seed}")


def collate_fn(batch):
    text_feats = [torch.as_tensor(item["text_feats"]) for item in batch]
    ids = torch.stack([item[0] for item in text_feats])
    mask = torch.stack([item[1] for item in text_feats])
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "video_feats": torch.stack([item["video_feats"] for item in batch]),
        "audio_feats": torch.stack([item["audio_feats"] for item in batch]),
        "label": torch.tensor([item["label_ids"] for item in batch]),
    }


def compute_accuracy(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            logits = model(**to_device(batch))[DENOISED_KEY]
            preds.extend(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())

    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_weighted": precision_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            labels, preds, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


@torch.enable_grad()
def compute_parameter_importance_converged(
    model, data_loader, device, num_batches=4
):
    model.eval()
    model.set_selection(False)
    bottlenecks = model.bottlenecks()
    accumulated = {
        key: torch.zeros(bottleneck.d_max, device=device)
        for key, bottleneck in bottlenecks.items()
    }
    valid_batches = 0

    for batch in islice(data_loader, int(num_batches)):
        labels = batch["label"].to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(**to_device(batch))
        loss, _ = compute_loss(outputs, labels)
        loss.backward()

        for key, bottleneck in bottlenecks.items():
            weight_grad = bottleneck.encoder.weight.grad
            if weight_grad is None:
                continue

            importance = (
                bottleneck.encoder.weight.detach().norm(dim=1) ** 2
            ) * (weight_grad.detach().norm(dim=1) ** 2)

            bias_grad = bottleneck.encoder.bias.grad
            if bias_grad is not None:
                importance = importance + (
                    bottleneck.encoder.bias.detach() ** 2
                ) * (bias_grad.detach() ** 2)

            decoder_grad = bottleneck.decoder.weight.grad
            if decoder_grad is not None:
                importance = importance + (
                    bottleneck.decoder.weight.detach().norm(dim=0) ** 2
                ) * (decoder_grad.detach().norm(dim=0) ** 2)
            accumulated[key] += torch.nan_to_num(importance)

        valid_batches += 1
        model.zero_grad(set_to_none=True)

    if valid_batches == 0:
        return 0

    for key, bottleneck in bottlenecks.items():
        score = accumulated[key] / valid_batches
        if float(score.sum()) <= 0:
            score = torch.ones_like(score)
        bottleneck._parameter_ranking.copy_(
            torch.argsort(score, descending=True)
        )
        bottleneck.rank_ready.fill_(1.0)
    model.set_selection(True)
    return len(bottlenecks)

def run_train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0

    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(**to_device(batch))
        loss, _ = compute_loss(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs[DENOISED_KEY].argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def train_and_validate(data_manager):
    print("=== Training Started ===")
    print(f"Current random seed: {args.seed}")
    print(f"Dataset: {args.dataset} | device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
    pretrain_path = os.path.join(args.save_dir, "pretrain_model.pth")
    print(f"Checkpoints will be written to: {checkpoint_path}")

    train_loader = make_loader(
        data_manager.mm_data["train"], args.train_batch_size, shuffle=True
    )
    val_loader = make_loader(
        data_manager.mm_data["dev"], args.eval_batch_size
    )

    model = MultiModalClassifier(args, num_classes=args.num_labels).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-9,
    )

    def lr_lambda(epoch):
        if epoch < 5:
            return 0.01 + (1 - 0.01) * epoch / 5
        progress = (epoch - 5) / (args.num_train_epochs - 5)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pretrain_epochs = int(args.pretrain_epochs)
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=20, delta=0.001)
    model.set_selection(False)
    print(f"IDAB pretrain phase: {pretrain_epochs} epochs without parameter selection")

    for epoch in range(args.num_train_epochs):
        train_loss, train_acc = run_train_epoch(
            model, train_loader, optimizer
        )
        model.eval()
        val_metrics = compute_accuracy(model, val_loader)
        scheduler.step()

        if epoch + 1 == pretrain_epochs:
            print("Computing global IDAB parameter importance on held-out dev batches...")
            ranked = compute_parameter_importance_converged(
                model,
                val_loader,
                device,
                num_batches=int(args.importance_batches),
            )
            model.eval()
            print(f"Ranked {ranked} IDAB bottlenecks; dynamic selection enabled")
            early_stopping = EarlyStopping(patience=20, delta=0.001)
            best_val_acc = 0.0

        selection_started = epoch + 1 > pretrain_epochs and all(
            float(b.rank_ready.item()) > 0 for b in model.bottlenecks().values()
        )
        if selection_started:
            early_stopping(val_metrics[args.select_metric])
            if early_stopping.early_stop:
                print("Early stopping triggered")
                print(f"Best validation checkpoint remains at {checkpoint_path}")
                break
            if val_metrics[args.select_metric] > best_val_acc:
                best_val_acc = val_metrics[args.select_metric]
                torch.save(model.state_dict(), checkpoint_path)
                print(
                    f"  new best val {args.select_metric} {best_val_acc:.4f} "
                    f"-> saved to {checkpoint_path}"
                )
        else:
            torch.save(model.state_dict(), pretrain_path)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | Val Acc: {val_metrics['accuracy']:.2%} | "
            f"Val F1: {val_metrics['f1_weighted']:.4f} | "
            f"Val F1m: {val_metrics['f1_macro']:.4f}"
        )


def test_model(data_manager):
    test_loader = make_loader(
        data_manager.mm_data["test"], args.test_batch_size
    )
    model = MultiModalClassifier(args, num_classes=args.num_labels).to(device)
    model_path = os.path.join(args.save_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return

    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    ready = model.enable_selection_if_ready()
    print(f"Dynamic IDAB selection enabled from checkpoint: {ready}")
    model.eval()
    print(f"Model loaded from: {model_path}")

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            outputs = model(**to_device(batch))
            preds = outputs[DENOISED_KEY].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    label_list = benchmarks[args.dataset]["labels"]
    print("\n=== Test Results ===")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(label_list))),
            target_names=label_list,
            digits=4,
            zero_division=0,
        )
    )

    test_metrics = compute_accuracy(model, test_loader)
    print("\n=== Comprehensive Evaluation Metrics ===")
    print(f"Accuracy (ACC): {test_metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {test_metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {test_metrics['recall_weighted']:.4f}")
    print(f"Weighted F1 Score: {test_metrics['f1_weighted']:.4f}")


def main():
    set_random_seed(args.seed)
    print("=== Dataset Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")

    data_dir = os.path.join(args.data_path, args.dataset)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data folder not found: {data_dir}")

    data = DataManager(args)
    print(
        f"Dataset loaded:\n"
        f"Training set: {len(data.mm_data['train'])} samples\n"
        f"Validation set: {len(data.mm_data['dev'])} samples\n"
        f"Test set: {len(data.mm_data['test'])} samples"
    )

    if args.test_only:
        test_model(data)
        return

    train_and_validate(data)
    test_model(data)


if __name__ == "__main__":
    main()
