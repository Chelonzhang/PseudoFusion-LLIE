import numpy as np
import torch
import cv2
from config import Config
from core_untils import numpy2tensor, tensor2numpy
from zero_shot_trainer import ZeroShotEnhancer

_enhancer = None


def get_enhancer():
    global _enhancer
    if _enhancer is None:
        _enhancer = ZeroShotEnhancer()
    return _enhancer


def enhance_pipeline(img_np):
    H, W = img_np.shape[:2]
    H_aligned = (H // 8) * 8
    W_aligned = (W // 8) * 8
    if H_aligned != H or W_aligned != W:
        img_np = cv2.resize(img_np, (W_aligned, H_aligned))

    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img_yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 1] = cv2.bilateralFilter(img_yuv[:, :, 1], 9, 75, 75)
    img_yuv[:, :, 2] = cv2.bilateralFilter(img_yuv[:, :, 2], 9, 75, 75)
    img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.float64) / 255.0

    img_t = numpy2tensor(img_np, Config.DEVICE)

    enhancer = get_enhancer()

    enhancer.optimize(img_t, num_steps=Config.ZERO_SHOT_STEPS)

    with torch.no_grad():
        enhancer.eval()
        I_final = enhancer.forward(img_t)[0]

    I_enhanced_np = tensor2numpy(I_final)

    gamma = 0.9
    I_gamma = (np.power(I_enhanced_np, gamma) * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(I_gamma, (0, 0), sigmaX=1.0)
    I_unsharp = cv2.addWeighted(I_gamma, 1.3, blurred, -0.3, 0)
    I_enhanced_np = np.clip(I_unsharp.astype(np.float64) / 255.0, 0.0, 1.0)

    return I_enhanced_np
