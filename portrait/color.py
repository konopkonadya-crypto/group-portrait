import cv2
import numpy as np


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _from_lab(lab):
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def transfer(patch, base_region, patch_mask, strength=1.0):
    """Подгоняет цвет и свет патча под участок эталона, на который он ляжет.

    strength: 0 — без изменений, 1 — полный перенос статистики Lab.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    inside = patch_mask > 127
    if strength == 0 or inside.sum() < 20 or base_region.shape[:2] != patch.shape[:2]:
        return patch

    src_lab = _lab(patch)
    dst_lab = _lab(base_region)
    out = src_lab.copy()
    for channel in range(3):
        src = src_lab[..., channel][inside]
        dst = dst_lab[..., channel][inside]
        mean_s, std_s = float(src.mean()), float(src.std())
        mean_d, std_d = float(dst.mean()), float(dst.std())
        ratio = 1.0 if std_s < 1e-3 else float(np.clip(std_d / std_s, 0.85, 1.18))
        adjusted = (src_lab[..., channel] - mean_s) * ratio + mean_d
        out[..., channel] = src_lab[..., channel] + strength * (adjusted - src_lab[..., channel])

    return _from_lab(out)
