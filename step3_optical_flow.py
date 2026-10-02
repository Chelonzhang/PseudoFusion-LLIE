import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from config import Config


class FlowAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        raft_model.eval()

        for p in raft_model.parameters():
            p.requires_grad = False

        self.raft = raft_model
        self.target_blur_sigma = 0.8

    def _rgb_to_grayscale_3ch(self, tensor_rgb):
        r, g, b = tensor_rgb[:, 0:1], tensor_rgb[:, 1:2], tensor_rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.repeat(1, 3, 1, 1)

    def compute_flow_between(self, target, source):
        B, C, H, W = target.shape
        t_norm = target * 2.0 - 1.0
        t_gray = self._rgb_to_grayscale_3ch(t_norm)
        t_blur = TF.gaussian_blur(t_gray, [3, 3], [self.target_blur_sigma, self.target_blur_sigma])
        s_norm = source * 2.0 - 1.0
        s_gray = self._rgb_to_grayscale_3ch(s_norm)
        s_blur = TF.gaussian_blur(s_gray, [3, 3], [self.target_blur_sigma, self.target_blur_sigma])
        return self.raft(t_blur, s_blur)[-1]

    def forward(self, frame_seq, denoise_flow=False, blur_kernel=5, blur_sigma=1.0, warp_seq=None):
        B, T, C, H, W = frame_seq.shape

        if warp_seq is None:
            warp_seq = frame_seq

        target = frame_seq[:, 0]
        output_target = warp_seq[:, 0]

        target_norm = target * 2.0 - 1.0
        target_gray = self._rgb_to_grayscale_3ch(target_norm)
        target_blurred = TF.gaussian_blur(target_gray, [3, 3],
                                           [self.target_blur_sigma, self.target_blur_sigma])

        if denoise_flow and blur_kernel > 0:
            blur_ks = min(blur_kernel, min(H, W) - 1) if min(H, W) % 2 == 0 else min(blur_kernel, min(H, W))
            blur_ks = blur_ks if blur_ks % 2 == 1 else blur_ks - 1
            if blur_ks >= 3:
                target_flow = TF.gaussian_blur(target_blurred, [blur_ks, blur_ks], [blur_sigma, blur_sigma])
            else:
                target_flow = target_blurred
        else:
            target_flow = target_blurred

        aligned_frames = []
        aligned_frames.append(output_target)

        for t in range(1, T):
            source = frame_seq[:, t]
            source_norm = source * 2.0 - 1.0
            source_gray = self._rgb_to_grayscale_3ch(source_norm)

            if denoise_flow and blur_kernel >= 3:
                source_flow = TF.gaussian_blur(source_gray, [blur_ks, blur_ks], [blur_sigma, blur_sigma])
            else:
                source_flow = source_gray

            flow = self.raft(target_flow, source_flow)[-1]

            flow_norm = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2 + 1e-8)
            flow[:, 0:1][flow_norm > 1.0] = 0.0
            flow[:, 1:2][flow_norm > 1.0] = 0.0
            flow_x = TF.gaussian_blur(flow[:, 0:1], [5, 5], [1.5, 1.5])
            flow_y = TF.gaussian_blur(flow[:, 1:2], [5, 5], [1.5, 1.5])
            flow = torch.cat([flow_x, flow_y], dim=1)
            flow_norm_sm = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2 + 1e-8)
            mn_f = flow_norm_sm.mean().item()
            th_sm = max(min(0.5 * mn_f, 0.15), 0.02)
            flow = flow * (flow_norm_sm >= th_sm).float()

            y, x = torch.meshgrid(
                torch.arange(H, device=Config.DEVICE),
                torch.arange(W, device=Config.DEVICE),
                indexing='ij'
            )
            vgrid = torch.stack((x, y), dim=0).float().unsqueeze(0) + flow
            vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
            vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
            vgrid = vgrid.permute(0, 2, 3, 1)

            source_warp = warp_seq[:, t]
            warped = F.grid_sample(source_warp, vgrid, mode='bicubic',
                                    padding_mode='border', align_corners=True)
            aligned_frames.append(warped)

        return torch.stack(aligned_frames, dim=1)
