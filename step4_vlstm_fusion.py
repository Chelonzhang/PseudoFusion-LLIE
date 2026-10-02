import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config
from vision_lstm2 import ViLBlockPair


def edge_gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def color_cosine_loss(pred, target):
    pred_norm = torch.norm(pred + 1e-6, dim=1, keepdim=True)
    target_norm = torch.norm(target + 1e-6, dim=1, keepdim=True)
    pred_norm = torch.clamp(pred_norm, min=1e-4)
    target_norm = torch.clamp(target_norm, min=1e-4)
    cos_sim = F.cosine_similarity(pred + 1e-6, target + 1e-6, dim=1)
    return torch.mean(1.0 - cos_sim)


def frequency_band_loss(pred, target, radius=0.65):
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred + 1e-8), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target + 1e-8), dim=(-2, -1))

    B, C, H, W = pred.size()
    cy, cx = H // 2, W // 2

    Y, X = torch.meshgrid(torch.arange(H, device=pred.device),
                          torch.arange(W, device=pred.device), indexing='ij')
    dist = torch.sqrt(((Y - cy) / (H / 2)) ** 2 + ((X - cx) / (W / 2)) ** 2)
    mask = (dist <= radius).float().unsqueeze(0).unsqueeze(0)

    abs_pred = torch.clamp(torch.abs(pred_fft) * mask, min=1e-8)
    abs_target = torch.clamp(torch.abs(target_fft) * mask, min=1e-8)

    return F.l1_loss(abs_pred, abs_target)


def spatial_consistency_loss(I_enh, I_input):
    enhance_ratio = (I_enh.detach() + 1e-6) / (I_input + 1e-6)

    dx = torch.abs(enhance_ratio[:, :, :, 1:] - enhance_ratio[:, :, :, :-1])
    dy = torch.abs(enhance_ratio[:, :, 1:, :] - enhance_ratio[:, :, :-1, :])

    edge_x = torch.exp(-10 * torch.abs(I_input[:, :, :, 1:] - I_input[:, :, :, :-1]))
    edge_y = torch.exp(-10 * torch.abs(I_input[:, :, 1:, :] - I_input[:, :, :-1, :]))

    return (dx * edge_x).mean() + (dy * edge_y).mean()


def exposure_loss(img, target_bright=0.6):
    gray = img.mean(dim=1, keepdim=True)
    pooled = F.avg_pool2d(gray, kernel_size=16, stride=16)
    return F.l1_loss(pooled, torch.ones_like(pooled) * target_bright)


def tv_loss_alpha(alpha):
    dx = torch.abs(alpha[:, :, :, :-1] - alpha[:, :, :, 1:]).mean()
    dy = torch.abs(alpha[:, :, :-1, :] - alpha[:, :, 1:, :]).mean()
    return dx + dy


def edge_aware_tv_loss(img, target):
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    weight_x = torch.exp(-10.0 * target_dx)
    weight_y = torch.exp(-10.0 * target_dy)

    img_dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    img_dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    return torch.mean(weight_x * img_dx) + torch.mean(weight_y * img_dy)


def compute_zero_shot_loss(I_enhanced, L, L_enhanced, R_clean, R_clean_ref, I_input, I_bright_target,
                           R=None, R_ref_frame=None, alpha=None, residual_img=None,
                           T_map=None, aligned_seq=None, R_fusion=None):
    cfg = Config

    I_enhanced = torch.clamp(I_enhanced, 0.0, 1.0)
    L = torch.clamp(L, 0.001, 1.0)
    L_enhanced = torch.clamp(L_enhanced, 0.001, 1.0)
    R_clean = torch.clamp(R_clean, 0.0, 1.0)
    R_clean_ref = torch.clamp(R_clean_ref, 0.0, 1.0)
    I_input = torch.clamp(I_input, 0.0, 1.0)
    I_bright_target = torch.clamp(I_bright_target, 0.0, 1.0)

    gray_I = torch.mean(I_input, dim=1, keepdim=True)
    blur_k = 5
    gray_I_pad = F.pad(gray_I, (blur_k//2,)*4, mode='replicate')
    L_target = F.avg_pool2d(gray_I_pad, kernel_size=blur_k, stride=1)
    loss_L_prior = F.l1_loss(L, L_target) * cfg.LOSS_L_PRIOR
    loss_L_prior = torch.where(torch.isfinite(loss_L_prior), loss_L_prior, torch.tensor(0.0, device=I_enhanced.device))

    R_gray_ref = R.mean(dim=1, keepdim=True) if R is not None else R_gray
    I_local = L * R_gray_ref
    I_smooth = F.avg_pool2d(I_local, kernel_size=5, stride=1, padding=2)
    W_noise = (cfg.W_NOISE_BASE + cfg.W_NOISE_STRENGTH * torch.exp(-3.0 * I_smooth)).detach()
    if T_map is not None and aligned_seq is not None and aligned_seq.size(1) >= 4:
        R_ref_frame = aligned_seq[:, 0].detach()
        R_fusion_smooth = F.avg_pool2d(R_fusion, kernel_size=5, stride=1, padding=2)
        target_flat = R_fusion_smooth.mean(dim=1, keepdim=True)
        target_texture = R_ref_frame.mean(dim=1, keepdim=True)
        target_consist = T_map * target_texture + (1 - T_map) * target_flat
        target_consist = target_consist.detach()
        R_gray = R_clean.mean(dim=1, keepdim=True)
        loss_consist = ((1 - T_map) * W_noise * torch.abs(R_gray - target_consist)).mean() * cfg.LOSS_CONSIST
    else:
        loss_consist = torch.tensor(0.0, device=I_enhanced.device)
    loss_consist = torch.where(torch.isfinite(loss_consist), loss_consist, torch.tensor(0.0, device=I_enhanced.device))

    if T_map is not None:
        T_3ch_s = T_map.expand(-1, 3, -1, -1).detach()
        w_flat = 1 - T_3ch_s
        W_noise_3ch = W_noise.expand(-1, 3, -1, -1)
        dx = torch.abs(R_clean[:, :, :, 1:] - R_clean[:, :, :, :-1])
        dy = torch.abs(R_clean[:, :, 1:, :] - R_clean[:, :, :-1, :])
        loss_sparse = ((w_flat[:, :, :, :-1] * W_noise_3ch[:, :, :, :-1] * dx).mean() +
                       (w_flat[:, :, :-1, :] * W_noise_3ch[:, :, :-1, :] * dy).mean()) * cfg.LOSS_SPARSE
    else:
        loss_sparse = torch.tensor(0.0, device=I_enhanced.device)
    loss_sparse = torch.where(torch.isfinite(loss_sparse), loss_sparse, torch.tensor(0.0, device=I_enhanced.device))

    if T_map is not None:
        T_3ch_t = T_map.expand(-1, 3, -1, -1).detach()
        dx = torch.abs(R_clean[:, :, :, 1:] - R_clean[:, :, :, :-1])
        dy = torch.abs(R_clean[:, :, 1:, :] - R_clean[:, :, :-1, :])
        loss_tv_texture = ((T_3ch_t[:, :, :, :-1] * torch.relu(0.03 - dx)).mean() +
                           (T_3ch_t[:, :, :-1, :] * torch.relu(0.03 - dy)).mean()) * cfg.LOSS_TV_TEXTURE
    else:
        loss_tv_texture = torch.tensor(0.0, device=I_enhanced.device)
    loss_tv_texture = torch.where(torch.isfinite(loss_tv_texture), loss_tv_texture, torch.tensor(0.0, device=I_enhanced.device))

    loss_bright = F.l1_loss(I_enhanced, I_bright_target) * cfg.LOSS_BRIGHT
    loss_bright = torch.where(torch.isfinite(loss_bright), loss_bright, torch.tensor(0.0, device=I_enhanced.device))
    loss_L_smooth = (
        torch.abs(L_enhanced[:, :, :, 1:] - L_enhanced[:, :, :, :-1]).mean() +
        torch.abs(L_enhanced[:, :, 1:, :] - L_enhanced[:, :, :-1, :]).mean()
    ) * cfg.LOSS_L_SMOOTH
    loss_L_smooth = torch.where(torch.isfinite(loss_L_smooth), loss_L_smooth, torch.tensor(0.0, device=I_enhanced.device))
    loss_color = color_cosine_loss(I_enhanced, I_bright_target) * cfg.LOSS_COLOR
    loss_color = torch.where(torch.isfinite(loss_color), loss_color, torch.tensor(0.0, device=I_enhanced.device))
    loss_freq = frequency_band_loss(I_enhanced, I_bright_target, radius=0.65) * cfg.LOSS_FREQ
    loss_freq = torch.where(torch.isfinite(loss_freq), loss_freq, torch.tensor(0.0, device=I_enhanced.device))

    dark_mask = (L < 0.3).float()
    if dark_mask.sum() > 100:
        R_masked = R_clean * dark_mask
        mean_rgb = R_masked.sum(dim=[2, 3]) / (dark_mask.sum() + 1e-6)
        loss_gray_world = ((mean_rgb[:, 0] - mean_rgb[:, 1]).abs().mean() +
                           (mean_rgb[:, 1] - mean_rgb[:, 2]).abs().mean() +
                           (mean_rgb[:, 0] - mean_rgb[:, 2]).abs().mean()) * cfg.LOSS_GRAY_WORLD
        loss_gray_world = torch.where(torch.isfinite(loss_gray_world), loss_gray_world, torch.tensor(0.0, device=I_enhanced.device))
    else:
        loss_gray_world = torch.tensor(0.0, device=I_enhanced.device)

    R_target = R.detach() if R is not None else None
    if R_target is not None:
        dev_rg_clean = R_clean[:, 0] - R_clean[:, 1]
        dev_rg_input = R_target[:, 0] - R_target[:, 1]
        dev_gb_clean = R_clean[:, 1] - R_clean[:, 2]
        dev_gb_input = R_target[:, 1] - R_target[:, 2]
        loss_color_deviation = (F.l1_loss(dev_rg_clean, dev_rg_input) +
                            F.l1_loss(dev_gb_clean, dev_gb_input)) * cfg.LOSS_COLOR_DEVIATION
        loss_color_deviation = torch.where(torch.isfinite(loss_color_deviation), loss_color_deviation, torch.tensor(0.0, device=I_enhanced.device))
    else:
        loss_color_deviation = torch.tensor(0.0, device=I_enhanced.device)

    if residual_img is not None:
        loss_residual_sparse = torch.mean(torch.abs(residual_img)) * cfg.LOSS_RESIDUAL_SPARSE
        loss_residual_sparse = torch.where(torch.isfinite(loss_residual_sparse), loss_residual_sparse, torch.tensor(0.0, device=I_enhanced.device))
    else:
        loss_residual_sparse = torch.tensor(0.0, device=I_enhanced.device)

    if residual_img is not None:
        res_mean_ch = residual_img.mean(dim=[2, 3])
        res_mean_gray = res_mean_ch.mean(dim=1, keepdim=True)
        loss_res_channel_align = (res_mean_ch - res_mean_gray).abs().mean() * cfg.LOSS_RES_CHANNEL_ALIGN
        loss_res_channel_align = torch.where(torch.isfinite(loss_res_channel_align), loss_res_channel_align, torch.tensor(0.0, device=I_enhanced.device))
    else:
        loss_res_channel_align = torch.tensor(0.0, device=I_enhanced.device)

    if alpha is not None:
        loss_exposure = exposure_loss(L_enhanced) * cfg.LOSS_EXPOSURE
        loss_exposure = torch.where(torch.isfinite(loss_exposure), loss_exposure, torch.tensor(0.0, device=I_enhanced.device))
        loss_tv_a = tv_loss_alpha(alpha) * cfg.LOSS_TV_ALPHA
        loss_tv_a = torch.where(torch.isfinite(loss_tv_a), loss_tv_a, torch.tensor(0.0, device=I_enhanced.device))
    else:
        loss_exposure = torch.tensor(0.0, device=I_enhanced.device)
        loss_tv_a = torch.tensor(0.0, device=I_enhanced.device)

    if R_target is not None:
        loss_bright_protect = F.relu(R_target.mean() - R_clean.mean()) * cfg.LOSS_BRIGHT_PROTECT
    else:
        loss_bright_protect = torch.tensor(0.0, device=I_enhanced.device)

    loss = (loss_L_prior + loss_bright + loss_L_smooth + loss_color + loss_freq +
            loss_consist + loss_sparse + loss_tv_texture +
            loss_gray_world + loss_color_deviation + loss_residual_sparse +
            loss_res_channel_align +
            loss_exposure + loss_tv_a + loss_bright_protect)

    loss_details = {
        'L_prior': loss_L_prior.item(),
        'bright': loss_bright.item(),
        'L_smooth': loss_L_smooth.item(),
        'color': loss_color.item(),
        'freq': loss_freq.item(),
        'consist': loss_consist.item(),
        'sparse': loss_sparse.item(),
        'tv_texture': loss_tv_texture.item(),
        'gray_world': loss_gray_world.item(),
        'color_deviation': loss_color_deviation.item(),
        'residual_sparse': loss_residual_sparse.item(),
        'res_channel_align': loss_res_channel_align.item(),
        'exposure': loss_exposure.item(),
        'tv_alpha': loss_tv_a.item(),
        'bright_protect': loss_bright_protect.item(),
        'total': loss.item(),
    }

    return loss, loss_details


class DeepViLSequenceModel(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=48, depth=1, seq_len=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.LeakyReLU(0.2),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.vil = ViLBlockPair(dim=hidden_dim, drop_path=0.0,
                                 conv_kind="causal1d", conv_kernel_size=3)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_dim, 3, 1, 1),
            nn.Tanh(),
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            last_conv = self.decoder[-2]
            last_conv.weight.zero_()
            nn.init.uniform_(last_conv.bias, -0.005, 0.005)
        with torch.no_grad():
            self.alpha_head[-2].weight.zero_()
            self.alpha_head[-2].bias.zero_()

    def forward(self, aligned_seq, base=None):
        B, T, C, H, W = aligned_seq.shape

        feats = []
        for t in range(T):
            f = self.encoder(aligned_seq[:, t])
            feats.append(f)

        feats_low = []
        for f in feats:
            f_low = self.downsample(f)
            feats_low.append(f_low)

        Bf, D, H_low, W_low = feats_low[0].shape

        feat_stack = torch.stack(feats_low, dim=1)
        feat_seq = feat_stack.permute(0, 3, 4, 1, 2).reshape(B * H_low * W_low, T, D)

        fused = self.vil(feat_seq)

        final_feat = fused[:, -1, :].view(B, H_low, W_low, D).permute(0, 3, 1, 2)

        final_feat_hr = self.upsample(final_feat)

        residual = self.decoder(final_feat_hr)
        with torch.no_grad():
            r_std = residual.std().clamp(min=0.01)
            scale = (0.15 / r_std).clamp(max=1.0)
        residual = residual * scale
        alpha_map = self.alpha_head(final_feat_hr)
        base_img = base if base is not None else aligned_seq[:, 0]
        R_clean = torch.clamp(base_img + residual, 0.0, 1.0)
        if not self.training:
            with torch.no_grad():
                base_mean = base_img.mean()
                clean_mean = R_clean.mean()
                base_ok = base_mean > 0.05
                clean_black = clean_mean < 0.02
                crushed = (base_mean - clean_mean) > 0.1
            if (not torch.isfinite(R_clean).all()) or (base_ok and clean_black and crushed):
                R_clean = torch.clamp(base_img, 0.0, 1.0)
                residual = torch.zeros_like(residual)

        return R_clean, residual, alpha_map
