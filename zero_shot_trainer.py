import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
from config import Config
from model_retinex import RUASRetinex, LightEnhancer, FusionLayer, EntropySharpener, PerFrameDenoiser
from step3_optical_flow import FlowAlignment
from step4_vlstm_fusion import DeepViLSequenceModel, compute_zero_shot_loss


def zero_dce_curve(L, alpha=1.0, n_iter=3):
    x = L
    for _ in range(n_iter):
        x = x + alpha * (x * x - x)
    return torch.clamp(x, 0.0, 1.0)


class ZeroShotEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.retinex = RUASRetinex(in_channels=3, iterations=3)
        self.flow_align = FlowAlignment()
        self.vil_model = DeepViLSequenceModel(
            in_dim=3,
            hidden_dim=Config.VIL_HIDDEN_DIM,
            depth=Config.VIL_DEPTH,
            seq_len=Config.VIL_SEQ_LEN
        )
        self.vil_residual = None
        self.vil_alpha = None
        self.l_enhancer = LightEnhancer()

        self.sharpener = EntropySharpener()

        self.fusion = FusionLayer()

        self.denoiser = PerFrameDenoiser(in_channels=3)

        self._init_states = {}
        for name in ['vil_model', 'l_enhancer', 'fusion', 'denoiser']:
            mod = getattr(self, name, None)
            if mod is not None:
                self._init_states[name] = {k: v.clone().detach().cpu() for k, v in mod.state_dict().items()}

        self._retinex_cache = None

        self.device = Config.DEVICE
        self.to(self.device)

    def forward(self, x):
        if self._retinex_cache is not None:
            R, L = self._retinex_cache
        else:
            R, L = self.retinex(x)
            if getattr(Config, 'RETINEX_METHOD', 'cnn') == 'srie':
                self._retinex_cache = (R, L)
        R = torch.nan_to_num(R, 0.0)
        L = torch.nan_to_num(L, 0.5)

        R_denoiser_in = R.detach().clone()
        R = self.denoiser(R)
        if not self.training:
            with torch.no_grad():
                in_mean = R_denoiser_in.mean()
                out_mean = R.mean()
                denoiser_crushed = (in_mean > 0.05) and (out_mean < in_mean * 0.3)
                if denoiser_crushed:
                    R = torch.clamp(R_denoiser_in, 0.0, 1.0)

        with torch.no_grad():
            gray_guide = torch.mean(x, dim=1, keepdim=True)
            R_smoothed = []
            for c in range(3):
                R_c = self.retinex.guided_filter(R[:, c:c+1], gray_guide, r=7)
                R_smoothed.append(R_c)
            R_ref = torch.cat(R_smoothed, dim=1)
            R_ref = torch.nan_to_num(R_ref, 0.0)

        with torch.no_grad():
            B, C, H, W = R.shape
            R_clean_ref = R_ref.clone()

            if Config.ENABLE_PSEUDO_SEQ:
                S_base = R_ref
                DAY2_ZERO_FLOW_TEST = False
                gray_guide_f = torch.mean(x, dim=1, keepdim=True)
                R_smoothed_f = []
                for c_f in range(C):
                    R_c_f = self.retinex.guided_filter(R[:, c_f:c_f+1], gray_guide_f, r=3)
                    R_smoothed_f.append(R_c_f)
                S_base_flow = torch.cat(R_smoothed_f, dim=1)
                var_local = F.avg_pool2d(R ** 2, kernel_size=3, stride=1, padding=1) - \
                            F.avg_pool2d(R, kernel_size=3, stride=1, padding=1) ** 2
                var_local = torch.relu(var_local)
                var_flat = var_local.view(B, C, -1).sort(dim=2)[0][:, :, :max(1, var_local.shape[2]*var_local.shape[3]//10)]
                sigma_est = torch.sqrt(var_flat.mean(dim=2, keepdim=True) + 1e-8).mean().item()
                if not math.isfinite(sigma_est) or sigma_est < 1e-6:
                    sigma_est = 8.0 / 255.0
                sigma_est = max(sigma_est, 3.0 / 255.0)

                frames = [R]
                for _ in range(2):
                    n = torch.randn_like(S_base) * sigma_est
                    frames.append(torch.clamp(S_base + n, 0.0, 1.0))
                sc = 0.5
                Hs, Ws = max(2, int(H * sc)), max(2, int(W * sc))
                aa_sg = max(0.3, (1.0 / sc - 1.0) * 0.5)
                aa_kz = max(3, int(aa_sg * 4 + 1))
                if aa_kz % 2 == 0: aa_kz += 1
                Sb = TF.gaussian_blur(S_base, [aa_kz, aa_kz], [aa_sg, aa_sg])
                dw = F.interpolate(Sb, size=(Hs, Ws), mode='bilinear', align_corners=True)
                up = F.interpolate(dw + torch.randn_like(dw) * sigma_est, size=(H, W), mode='bilinear', align_corners=True)
                frames.append(torch.clamp(up, 0.0, 1.0))
                pseudo_seq = torch.stack(frames, dim=1)

                if Config.ENABLE_FLOW:
                    if DAY2_ZERO_FLOW_TEST:
                        y_f, x_f = torch.meshgrid(torch.arange(H, device=R.device), torch.arange(W, device=R.device), indexing='ij')
                        zero_grid = torch.stack((x_f, y_f), dim=0).float().unsqueeze(0)
                        zero_grid[:, 0] = 2.0 * zero_grid[:, 0] / max(W - 1, 1) - 1.0
                        zero_grid[:, 1] = 2.0 * zero_grid[:, 1] / max(H - 1, 1) - 1.0
                        zero_grid = zero_grid.permute(0, 2, 3, 1)
                        f3_zero = F.grid_sample(pseudo_seq[:, 3], zero_grid, mode='bicubic', padding_mode='border', align_corners=True)
                        gR_f = S_base_flow.mean(dim=1, keepdim=True)
                        gf = (torch.abs(torch.gradient(gR_f, dim=3)[0])+torch.abs(torch.gradient(gR_f, dim=2)[0]))/2
                        tm = torch.sigmoid((gf-0.01)/0.005)
                        f3_direct = pseudo_seq[:, 3]
                        pseudo_seq[:, 3] = torch.clamp(tm*f3_zero+(1-tm)*f3_direct, 0.0, 1.0)
                    else:
                        flow_ref = S_base_flow
                        sc_f = 0.5; Hs_f = max(2, int(H * sc_f)); Ws_f = max(2, int(W * sc_f))
                        aa_sf = max(0.3, (1.0/sc_f-1.0)*0.5); aa_kf = max(3, int(aa_sf*4+1))
                        if aa_kf%2==0: aa_kf+=1
                        Sbf = TF.gaussian_blur(S_base_flow, [aa_kf, aa_kf], [aa_sf, aa_sf])
                        fsrc = F.interpolate(Sbf, (Hs_f, Ws_f), mode='bilinear', align_corners=True)
                        fsrc_up = F.interpolate(fsrc, (H, W), mode='bilinear', align_corners=True)
                        fseq2 = torch.stack([flow_ref, fsrc_up], dim=1)
                        aligned_f3 = self.flow_align(fseq2, warp_seq=torch.stack([pseudo_seq[:,0], pseudo_seq[:,3]], dim=1))
                        f3_flow = aligned_f3[:, 1]
                        gR_f = S_base_flow.mean(dim=1, keepdim=True)
                        gf = (torch.abs(torch.gradient(gR_f, dim=3)[0])+torch.abs(torch.gradient(gR_f, dim=2)[0]))/2
                        tm = torch.sigmoid((gf-0.01)/0.005)
                        f3_direct = pseudo_seq[:, 3]
                        pseudo_seq[:, 3] = torch.clamp(tm*f3_flow+(1-tm)*f3_direct, 0.0, 1.0)
                    aligned_seq = pseudo_seq
                else:
                    aligned_seq = pseudo_seq
                aligned_seq = torch.nan_to_num(aligned_seq, 0.0)

                if Config.ENABLE_FUSION:
                    gf_r = 5
                    low_list, high_list = [], []
                    for t_idx in range(aligned_seq.size(1)):
                        ft = aligned_seq[:, t_idx]
                        lt = torch.zeros_like(ft)
                        for cc in range(C):
                            lt[:, cc:cc+1] = self.retinex.guided_filter(ft[:, cc:cc+1], gray_guide, r=gf_r)
                        low_list.append(lt)
                        high_list.append(ft - lt)

                    R_low_base = (low_list[1] + low_list[2] + low_list[3]) / 3.0

                    gR_t = S_base_flow.mean(dim=1, keepdim=True)
                    gf_t = (torch.abs(torch.gradient(gR_t, dim=3)[0]) + torch.abs(torch.gradient(gR_t, dim=2)[0])) / 2
                    T_w = torch.sigmoid((gf_t - 0.02) / 0.01)
                    w_avg = 0.2 + 0.8 * (1 - T_w)

                    high_avg = (high_list[1] + high_list[2] + high_list[3]) / 3.0
                    high_median = torch.median(torch.stack([high_list[1], high_list[2], high_list[3]]), dim=0).values
                    R_high_fused = w_avg * (T_w * high_avg + (1 - T_w) * high_median) + (1 - w_avg) * high_list[1]

                    R_fusion = torch.clamp(R_low_base + R_high_fused, 0.0, 1.0)
                    self.T_map = T_w
                    self.aligned_seq = aligned_seq.detach()
                    self.R_fusion = R_fusion
                else:
                    R_fusion = aligned_seq.mean(dim=1)
                    gR_fb = aligned_seq[:, 0].mean(dim=1, keepdim=True)
                    gf_fb = (torch.abs(torch.gradient(gR_fb, dim=3)[0]) + torch.abs(torch.gradient(gR_fb, dim=2)[0])) / 2
                    self.T_map = torch.sigmoid((gf_fb - 0.01) / 0.005)
                    self.aligned_seq = aligned_seq.detach()
                    self.R_fusion = R_fusion
            else:
                R_fusion = R
                self.T_map = torch.ones_like(R[:, :1]) * 0.5
                self.aligned_seq = None
                self.R_fusion = R_fusion

        if Config.ENABLE_VIL and Config.ENABLE_PSEUDO_SEQ:
            with torch.set_grad_enabled(self.training):
                vil_in = torch.cat([R_fusion.unsqueeze(1), aligned_seq], dim=1)
                R_clean, self.vil_residual, self.vil_alpha = self.vil_model(vil_in, base=R_fusion)
                R_clean = torch.nan_to_num(R_clean, 0.0)
                self.vil_residual = torch.nan_to_num(self.vil_residual, 0.0)
                self.vil_alpha = torch.nan_to_num(self.vil_alpha, 0.0)
        else:
            R_clean = R_fusion

        R_sharpened = self.sharpener(R_clean)
        if getattr(self, 'vil_alpha', None) is not None and Config.ENABLE_VIL:
            alpha_use = self.vil_alpha
            if not self.training and Config.VIL_ALPHA_SCALE != 1.0:
                alpha_use = alpha_use * Config.VIL_ALPHA_SCALE
            L_enhanced = zero_dce_curve(L, alpha=alpha_use, n_iter=Config.L_ENHANCE_ITERS)
        else:
            alpha_use = None
            L_enhanced = torch.clamp(L ** Config.L_ENHANCE_GAMMA, 0.0, 1.0)
        I_enhanced = self.fusion(R_sharpened, L_enhanced)
        alpha = alpha_use

        return I_enhanced, R_clean, L, L_enhanced, R, R_clean_ref, alpha, self.vil_residual, self.T_map, self.aligned_seq, self.R_fusion

    def _build_bright_target(self, x):
        I_bright = torch.clamp(x ** 0.25, 0.0, 1.0)
        return I_bright.detach()

    def optimize(self, img_tensor, num_steps=None):
        if num_steps is None:
            num_steps = Config.ZERO_SHOT_STEPS

        trainable_params = (
            list(self.fusion.parameters()) +
            list(self.denoiser.parameters())
        )
        if Config.ENABLE_VIL:
            trainable_params = trainable_params + list(self.vil_model.parameters())

        optimizer = optim.AdamW(trainable_params, lr=Config.LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=Config.LR_MIN
        )

        I_bright_target = self._build_bright_target(img_tensor)

        for name in ['vil_model', 'l_enhancer', 'fusion', 'denoiser']:
            mod = getattr(self, name, None)
            if mod is not None and name in self._init_states:
                mod.load_state_dict({k: v.to(self.device) for k, v in self._init_states[name].items()})

        self._retinex_cache = None

        self.train()

        best_loss = float('inf')
        best_result = None
        history = {'total': [], 'L_prior': [], 'bright': [],
                   'L_smooth': [], 'color': [], 'freq': [],
                   'consist': [], 'sparse': [], 'tv_texture': [],
                   'gray_world': [], 'color_deviation': [], 'residual_sparse': [],
                   'exposure': [], 'tv_alpha': [], 'bright_protect': []}

        for step in range(num_steps):
            optimizer.zero_grad()

            I_enhanced, R_clean, L, L_enhanced, R_input, R_clean_ref, alpha, residual, T_map, aligned_seq, R_fusion = self.forward(img_tensor)

            loss, details = compute_zero_shot_loss(
                I_enhanced, L, L_enhanced, R_clean, R_clean_ref, img_tensor, I_bright_target,
                R=R_input, alpha=alpha, residual_img=residual,
                T_map=T_map, aligned_seq=aligned_seq, R_fusion=R_fusion
            )

            if not torch.isfinite(loss):
                for k, v in details.items():
                    if k in history:
                        history[k].append(float('nan'))
                last_result = I_enhanced.detach().clone()
                continue

            loss.backward()

            for p in trainable_params:
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, 0.0)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            for k, v in details.items():
                if k in history:
                    history[k].append(v)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_result = I_enhanced.detach().clone()

            last_result = I_enhanced.detach().clone()

            if (step + 1) % 50 == 0:
                print(f"  [train] {step+1}/{num_steps}")

        self.eval()

        if best_result is not None:
            return best_result, history
        else:
            return last_result, history
