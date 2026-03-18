from torch import nn
import torch.nn.functional as F

def saliency(x, y):
    p_fused = F.softmax(x, dim=1)
    p_denoised = F.softmax(y, dim=1)
    saliency_loss = F.kl_div(
        p_denoised.log(),
        p_fused,     
        reduction='batchmean'
    )

    return saliency_loss

def compute_loss(outputs, labels, alpha=0.8, beta=0.8, denoised=True, deredundant=True):
    loss_dict = {
        'loss_fused': 0.0,
        'loss_text': 0.0,
        'loss_denoised': 0.0,
        'saliency_denoise': 0.0,
        'saliency_text': 0.0,
        'saliency_video': 0.0,
        'saliency_audio': 0.0,
        'total_loss': 0.0
    }

    loss_fused = F.cross_entropy(outputs['fused_preds'], labels)
    loss_text = F.cross_entropy(outputs['text_preds'], labels)
    loss_necessary = loss_fused + loss_text
    total_loss = loss_necessary

    loss_dict.update({
        'loss_fused': loss_fused,
        'loss_text': loss_text,
        'total_loss': total_loss
    })

    if denoised and outputs['denoised_feats'] is not None:
        loss_denoised = F.cross_entropy(outputs['denoised_preds'], labels)
        saliency_denoise = saliency(outputs['fused_feats'], outputs['denoised_feats'])
        denoised_loss = alpha * saliency_denoise + loss_denoised
        total_loss += denoised_loss
        
        loss_dict.update({
            'loss_denoised': loss_denoised,
            'saliency_denoise': saliency_denoise
        })

    if deredundant and outputs['denoised_text_feats'] is not None:
        saliency_loss_text = saliency(outputs['text_feats'], outputs['denoised_text_feats'])
        saliency_loss_video = 0.0
        saliency_loss_audio = 0.0
        
        if outputs['denoised_video_feats'] is not None:
            saliency_loss_video = saliency(outputs['video_feats'], outputs['denoised_video_feats'])
        if outputs['denoised_audio_feats'] is not None:
            saliency_loss_audio = saliency(outputs['audio_feats'], outputs['denoised_audio_feats'])
        
        deredundant_loss = saliency_loss_text + (saliency_loss_video + saliency_loss_audio) * 0.8
        total_loss += deredundant_loss
        
        loss_dict.update({
            'saliency_text': saliency_loss_text,
            'saliency_video': saliency_loss_video,
            'saliency_audio': saliency_loss_audio,
            'total_loss': total_loss
        })

    return total_loss, loss_dict
