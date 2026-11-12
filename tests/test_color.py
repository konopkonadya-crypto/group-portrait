import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portrait.color import transfer


def solid(rgb, size=80):
    return np.full((size, size, 3), rgb, np.uint8)


def test_strength_zero_keeps_patch():
    patch = solid((200, 120, 110))
    base = solid((120, 150, 180))
    mask = np.zeros((80, 80), np.uint8)
    mask[10:70, 10:70] = 255
    assert np.array_equal(transfer(patch, base, mask, 0.0), patch)


def test_full_strength_reaches_base_color():
    patch = solid((200, 120, 110))
    base = solid((120, 150, 180))
    mask = np.zeros((80, 80), np.uint8)
    mask[10:70, 10:70] = 255
    out = transfer(patch, base, mask, 1.0)
    assert np.abs(out[40, 40].astype(int) - np.array([120, 150, 180])).max() <= 8


def test_half_strength_is_between():
    patch = solid((200, 120, 110))
    base = solid((120, 150, 180))
    mask = np.zeros((80, 80), np.uint8)
    mask[10:70, 10:70] = 255
    half = transfer(patch, base, mask, 0.5)[40, 40].astype(int)
    full = transfer(patch, base, mask, 1.0)[40, 40].astype(int)
    original = patch[40, 40].astype(int)
    assert np.abs(half - original).sum() < np.abs(full - original).sum()
    assert np.abs(half - original).sum() > 0


def test_different_tints_converge():
    base = solid((150, 100, 90))
    mask = np.zeros((80, 80), np.uint8)
    mask[10:70, 10:70] = 255
    cool = transfer(solid((120, 95, 130)), base, mask, 1.0)[40, 40].astype(int)
    warm = transfer(solid((180, 90, 70)), base, mask, 1.0)[40, 40].astype(int)
    assert np.abs(cool - warm).max() <= 10


def test_tiny_mask_returns_patch():
    patch = solid((200, 120, 110))
    base = solid((120, 150, 180))
    mask = np.zeros((80, 80), np.uint8)
    mask[40, 40] = 255
    assert np.array_equal(transfer(patch, base, mask, 1.0), patch)


def test_shape_mismatch_is_safe():
    patch = np.zeros((80, 80, 3), np.uint8)
    base = np.zeros((40, 80, 3), np.uint8)
    mask = np.full((80, 80), 255, np.uint8)
    assert transfer(patch, base, mask, 1.0).shape == patch.shape
