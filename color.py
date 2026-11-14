import cv2
import numpy as np


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _from_lab(lab):
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _guide(image):
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY)


def _lowfreq(gray, radius):
    """Низкие частоты яркости: освещение без текстуры."""
    if hasattr(cv2, "ximgproc"):
        try:
            return cv2.ximgproc.guidedFilter(gray, gray, int(radius), 1e-3)
        except cv2.error:
            pass
    size = int(radius) | 1
    return cv2.GaussianBlur(gray, (size, size), 0)


def harmonize(patch, base_region, patch_mask, strength=1.0, ring=None):
    """Вписывает патч в окружение: цвет, затем локальный свет по остатку.

    Свет считается только как остаток после цветового сдвига, поэтому
    на ровных участках результат совпадает с эталоном без перекоса.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    inside = patch_mask > 127
    if strength <= 0 or inside.sum() < 20 or base_region.shape[:2] != patch.shape[:2]:
        return patch

    h, w = patch.shape[:2]
    radius = max(8, min(h, w) // 12)

    src_lab = _lab(patch)
    dst_lab = _lab(base_region)

    src_mean = np.array([src_lab[..., c][inside].mean() for c in range(3)], np.float32)
    ring_mask = None
    if ring is not None:
        ring_mask = (ring > 127) & ~inside
    if ring_mask is None or ring_mask.sum() < 50:
        ring_mask = None
    dst_mean = np.array([
        (dst_lab[..., c][ring_mask].mean() if ring_mask is not None else dst_lab[..., c][inside].mean())
        for c in range(3)
    ], np.float32)

    # 1. Цвет и общая яркость: полный перенос средних значений Lab.
    patch = _shift_lab(src_lab, dst_mean - src_mean, strength)

    # 2. Локальный свет: остаток низких частот после сдвига.
    src_l = _lowfreq(_guide(patch), radius)
    dst_l = _lowfreq(_guide(base_region), radius)
    residual = (dst_l - src_l) * 255.0
    if residual.size:
        patch = np.clip(patch.astype(np.float32) + residual[..., None] * strength, 0, 255).astype(np.uint8)

    # 3. Микроконтраст эталона: немного возвращаем текстуру окружения.
    detail_dst = dst_l - cv2.GaussianBlur(dst_l, (0, 0), 2.0)
    detail_src = src_l - cv2.GaussianBlur(src_l, (0, 0), 2.0)
    patch = np.clip(
        patch.astype(np.float32) + ((detail_dst - detail_src) * strength * 0.3)[..., None] * 255.0, 0, 255
    ).astype(np.uint8)

    return patch


def _shift_lab(src_lab, shift, strength):
    lab = src_lab.copy()
    lab[..., 0] += shift[0] * strength
    lab[..., 1] += shift[1] * strength
    lab[..., 2] += shift[2] * strength
    return _from_lab(lab)


# прежнее имя оставлено для совместимости
transfer = harmonize
