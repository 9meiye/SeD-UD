"""Total SeD-UD loss based on Equations 19-21."""

import torch.nn.functional as F


def saliency(p_logits, q_logits):
    """Computes D_KL(p || q) from classifier logits."""
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    return F.kl_div(q.log(), p, reduction="batchmean")


def compute_loss(outputs, labels):
    l_fu = F.cross_entropy(outputs["fused_preds"], labels)
    l_dn = saliency(
        outputs["fused_preds"], outputs["denoised_preds"]
    ) + F.cross_entropy(outputs["denoised_preds"], labels)

    l_dr_t = saliency(
        outputs["text_preds"], outputs["denoised_text_preds"]
    )
    l_dr_v = saliency(
        outputs["video_preds"], outputs["denoised_video_preds"]
    )
    l_dr_a = saliency(
        outputs["audio_preds"], outputs["denoised_audio_preds"]
    )

    lambda_t, lambda_v, lambda_a = 1.0, 0.8, 0.8
    eta, omega = 0.8, 1.0
    l_dr = (
        lambda_t * l_dr_t
        + lambda_v * l_dr_v
        + lambda_a * l_dr_a
    ) / 3.0
    total = l_dr + eta * l_dn + omega * l_fu

    return total, {
        "loss_fused": l_fu,
        "loss_denoised": l_dn,
        "loss_dr_text": l_dr_t,
        "loss_dr_video": l_dr_v,
        "loss_dr_audio": l_dr_a,
        "loss_de_redundancy": l_dr,
        "total_loss": total,
    }
