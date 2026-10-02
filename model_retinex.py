import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ReflectanceDenoiseBlock(nn.Module):
    def __init__(self, channel=3):
        super().__init__()
        self.act = nn.LeakyReLU(0.05)
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv_fuse = nn.Conv2d(channel * 2, channel, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        r1 = self.act(self.conv1(x) + x)
        d1 = self.act(self.conv2(r1))
        r2 = self.act(self.conv3(r1))
        d2 = self.act(self.conv4(r2))
        out = torch.cat([d1, d2], dim=1)
        return self.conv_fuse(out)


class RUASRetinex(nn.Module):
    def __init__(self, in_channels=3, iterations=3):
        super().__init__()
        self.iterations = iterations

        self.l_net = ReflectanceDenoiseBlock(channel=3)
        self.l_out = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid()
        )

        self._eng = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_engine(self):
        if self._eng is None:
            import matlab.engine
            self._eng = matlab.engine.start_matlab()
            import os
            srie_dir = os.path.dirname(os.path.abspath(__file__))
            self._eng.addpath(srie_dir)
        return self._eng

    def _srie_forward(self, x):
        import numpy as np
        import matlab
        eng = self._get_engine()

        B, C, H, W = x.shape
        R_list, L_list = [], []
        for b in range(B):
            img_np = x[b].detach().cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            img_ml = matlab.double(img_np.tolist())
            R_ml, L_ml = eng.srie_wrapper(img_ml, nargout=2)
            R_np = np.array(R_ml)
            L_np = np.array(L_ml)
            if L_np.ndim == 3:
                L_np = L_np[:, :, 0]
            R_list.append(torch.from_numpy(R_np).permute(2, 0, 1).float())
            L_list.append(torch.from_numpy(L_np).unsqueeze(0).float())

        R = torch.stack(R_list).to(x.device)
        L = torch.stack(L_list).to(x.device)
        L = torch.clamp(L, min=0.05)
        R = torch.clamp(R, 0.02, 1.0)
        if Config.USE_PRE_SMOOTH:
            R = self._denoise_r(R, L)

        return R, L

    def _denoise_r(self, R, L, r=5, eps=1e-2):
        B, C, H, W = R.shape
        R_gray = R.mean(dim=1, keepdim=True)

        R_smooth = torch.zeros_like(R)
        for c in range(C):
            R_smooth[:, c:c+1] = self.guided_filter(R[:, c:c+1], R_gray, r=r)

        R_median = torch.zeros_like(R)
        for c in range(C):
            patches = F.unfold(R[:, c:c+1], kernel_size=5, padding=2)
            med = patches.median(dim=1).values
            R_median[:, c:c+1] = med.view(B, 1, H, W)

        w_smooth = torch.clamp(0.5 + 0.8 * torch.exp(-8.0 * R_gray), 0.0, 1.0).detach()
        R_out = w_smooth * R_median + (1 - w_smooth) * R_smooth
        return torch.clamp(R_out, 0.0, 1.0)

    def max_operation(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def guided_filter(self, L, guide, r=3):
        assert r % 2 == 1, "r must be odd"
        mean_L = F.avg_pool2d(L, kernel_size=r, stride=1, padding=r//2)
        mean_g = F.avg_pool2d(guide, kernel_size=r, stride=1, padding=r//2)
        corr_Lg = F.avg_pool2d(L * guide, kernel_size=r, stride=1, padding=r//2)
        corr_gg = F.avg_pool2d(guide * guide, kernel_size=r, stride=1, padding=r//2)

        cov_Lg = corr_Lg - mean_L * mean_g
        var_g = corr_gg - mean_g * mean_g

        eps = 1e-6
        a = cov_Lg / (var_g + eps)
        b = mean_L - a * mean_g

        mean_a = F.avg_pool2d(a, kernel_size=r, stride=1, padding=r//2)
        mean_b = F.avg_pool2d(b, kernel_size=r, stride=1, padding=r//2)

        return torch.clamp(mean_a * guide + mean_b, 0.0, 1.0)

    def forward(self, x):
        if getattr(Config, 'RETINEX_METHOD', 'cnn') == 'srie':
            return self._srie_forward(x)

        u = torch.ones_like(x)
        gray = torch.mean(x, dim=1, keepdim=True)

        for k in range(self.iterations):
            if k == 0:
                t_hat = self.max_operation(x)
            else:
                t_hat = self.max_operation(u) - 0.5 * (u - x)

            feat = self.l_net(t_hat)
            L = self.l_out(feat)
            L = torch.clamp(L, 0.001, 1.0)

            u = torch.clamp(x / L, 0.0, 1.0)

        return u, L


class LightEnhancer(nn.Module):
    def __init__(self, n_iter=8):
        super().__init__()
        self.n_iter = n_iter
        self.curve_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, L):
        alpha = self.curve_net(L) * 2.0
        L_enhanced = L
        for _ in range(self.n_iter):
            L_enhanced = L_enhanced + alpha * L_enhanced * (1.0 - L_enhanced)
        return torch.clamp(L_enhanced, 0.0, 1.0), alpha


class FusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.fuse[-2].weight.zero_()
            self.fuse[-2].bias.zero_()

    def forward(self, R_clean, L_enhanced):
        L_3ch = L_enhanced.expand(-1, 3, -1, -1)
        inp = torch.cat([R_clean, L_enhanced], dim=1)
        weight = self.fuse(inp)
        I_fused = weight * R_clean + (1 - weight) * (R_clean * L_3ch)
        return torch.clamp(I_fused, 0.0, 1.0)


class PerFrameDenoiser(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, 3, 1, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.net[-1]:
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = self.net(x)
        residual = torch.clamp(residual, min=-0.5 * x)
        return torch.clamp(x + residual, 0.0, 1.0)


class EntropySharpener(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 0.3

    def forward(self, R_clean):
        R_smooth = F.avg_pool2d(R_clean, kernel_size=7, stride=1, padding=3)

        detail = R_clean - R_smooth

        mean_local = F.avg_pool2d(R_clean, kernel_size=3, stride=1, padding=1)
        diff_local = R_clean - mean_local
        var_local = F.avg_pool2d(diff_local ** 2, kernel_size=3, stride=1, padding=1)
        weight = torch.sqrt(var_local + 1e-8)
        weight = torch.clamp(weight, min=1e-6)
        w_min = weight.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        w_max = weight.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        weight = (weight - w_min) / (w_max - w_min + 0.01) * 0.7 + 0.3

        gray_weight = torch.mean(weight, dim=1, keepdim=True)

        R_sharpened = R_clean + self.gamma * gray_weight * detail
        return torch.clamp(R_sharpened, 0.0, 1.0)
