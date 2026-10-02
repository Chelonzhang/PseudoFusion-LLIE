import torch
import numpy as np
import cv2


def numpy2tensor(np_arr, device):
    return torch.tensor(np_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor2numpy(t_arr):
    return t_arr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def save_img(path, img_np):
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
