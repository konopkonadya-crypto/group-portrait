import cv2
import numpy as np

from .color import transfer
from .face import ANCHORS, NOSE_TIP

CROP_MARGIN = 60
EDGE_BAND = 24
FACE_RATIO = 1.2
FACE_SHARE = 0.3
SMART_EDGES = True


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def soft_alpha(mask, feather):
    if feather <= 0:
        return (mask > 127).astype(np.float32)
    k = int(feather) * 2 + 1
    return np.clip(cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0, 0, 1)


def refine_edge(rgb, mask, band=EDGE_BAND):
    box = bbox_of(mask)
    if box is None:
        return mask
    h, w = mask.shape
    x0, y0 = max(0, box[0] - band), max(0, box[1] - band)
    x1, y1 = min(w, box[2] + band), min(h, box[3] + band)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    local = mask[y0:y1, x0:x1]
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    near = cv2.dilate(local, band_kernel)
    sure_fg = cv2.erode(local, np.ones((5, 5), np.uint8))
    gc = np.full(local.shape, cv2.GC_BGD, np.uint8)
    gc[near > 0] = cv2.GC_PR_BGD
    gc[local > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    refined = np.zeros_like(mask)
    refined[y0:y1, x0:x1] = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return refined if refined.any() else mask



def smart_matte(rgb, mask, feather=12, iterations=3):
    """Уточняет границу по содержимому снимка, учитывая волосы и края одежды.

    Маска расширяется в неизвестную полосу, внутри полосы альфа считается по
    близости цвета к переднему и заднему плану, затем сглаживается guided filter.
    """
    box = bbox_of(mask)
    if box is None:
        return mask
    h, w = mask.shape
    margin = max(12, int(feather) * 2)
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(w, box[2] + margin), min(h, box[3] + margin)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    local = (mask[y0:y1, x0:x1] > 127).astype(np.uint8)

    core = cv2.erode(local, np.ones((3, 3), np.uint8), iterations=1)
    unknown = cv2.dilate(local, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8))
    unknown = (unknown > 0) & (core == 0)
    if core.sum() < 30 or unknown.sum() < 30:
        return mask

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_mean = lab[core > 0].mean(axis=0)
    bg_ring = cv2.dilate(local, np.ones((margin * 3 + 1, margin * 3 + 1), np.uint8))
    bg_sel = (bg_ring > 0) & (local == 0)
    if bg_sel.sum() < 30:
        return mask
    bg_mean = lab[bg_sel].mean(axis=0)

    dfg = np.linalg.norm(lab - fg_mean, axis=2)
    dbg = np.linalg.norm(lab - bg_mean, axis=2)
    alpha = np.clip(dbg / (dfg + dbg + 1e-6), 0, 1)
    for _ in range(max(1, iterations)):
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, margin / 3.0))

    guided = alpha.astype(np.float32)
    if hasattr(cv2, "ximgproc"):
        try:
            guided = cv2.ximgproc.guidedFilter(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), guided.astype(np.float32),
                                               max(4, margin), 1e-3)
        except cv2.error:
            pass

    alpha = np.clip(guided, 0, 1)
    alpha[core > 0] = 1.0
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = alpha
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def face_span(face):
    return float(np.linalg.norm(face[ANCHORS[1]] - face[ANCHORS[0]]))


def share_inside(face, mask):
    h, w = mask.shape
    pts = np.round(face).astype(int)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    hits = np.zeros(len(pts), bool)
    hits[ok] = mask[pts[ok, 1], pts[ok, 0]] > 127
    return float(hits.mean())


def estimate_alignment(donor_face, base_face):
    src = np.float64(donor_face[ANCHORS])
    dst = np.float64(base_face[ANCHORS])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None or int(inliers.sum()) < 5:
        return None
    return matrix


def _mul(a, b):
    return (np.vstack([a, [0, 0, 1]]) @ np.vstack([b, [0, 0, 1]]))[:2]


def _order_match(chosen, work_faces):
    """Запасное сопоставление: слева направо по относительному положению в кадре."""
    order_donor = sorted(range(len(chosen)), key=lambda i: chosen[i][NOSE_TIP][0])
    order_work = sorted(range(len(work_faces)), key=lambda j: work_faces[j][NOSE_TIP][0])
    matches = {}
    for slot, i in enumerate(order_donor):
        if slot < len(order_work):
            matches[i] = order_work[slot]
    return matches


def plan_faces(selection, donor_faces, work_faces, offset):
    """Лица донора внутри выделения и их пары с лицами рабочего снимка, без повторов."""
    chosen = [face for face in donor_faces if share_inside(face, selection) >= FACE_SHARE]
    if not chosen or not work_faces:
        return chosen, {}

    pairs = []
    for i, face in enumerate(chosen):
        guess = face[NOSE_TIP] + np.asarray(offset, np.float64)
        for j, target in enumerate(work_faces):
            dist = float(np.linalg.norm(target[NOSE_TIP] - guess))
            if dist <= FACE_RATIO * face_span(target):
                pairs.append((dist, i, j))
    matches, used = {}, set()
    for _, i, j in sorted(pairs):
        if i in matches or j in used:
            continue
        matches[i] = j
        used.add(j)

    # Если кто-то остался без пары, добираем по порядку слева направо.
    if len(matches) < len(chosen):
        spare = {i: j for i, j in _order_match(chosen, work_faces).items()
                 if i not in matches and j not in used}
        for i, j in spare.items():
            matches[i] = j
            used.add(j)
    return chosen, matches


def split_selection(selection, centers):
    """Делит выделение между лицами: каждый пиксель достаётся ближайшему лицу."""
    if len(centers) <= 1:
        return [selection]
    ys, xs = np.nonzero(selection)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    dist = np.stack([np.linalg.norm(pts - np.asarray(c, np.float32), axis=1) for c in centers], axis=1)
    label = dist.argmin(axis=1)
    parts = []
    for k in range(len(centers)):
        part = np.zeros_like(selection)
        chosen = label == k
        part[ys[chosen], xs[chosen]] = 255
        parts.append(part)
    return parts


def _blend_poisson(base, patch, patch_mask, ox, oy):
    hard = (patch_mask > 127).astype(np.uint8) * 255
    hard[:2, :] = hard[-2:, :] = 0
    hard[:, :2] = hard[:, -2:] = 0
    box = bbox_of(hard)
    if box is None:
        return None
    cx0, cy0, cx1, cy1 = box
    bh, bw = base.shape[:2]
    ax0, ay0, ax1, ay1 = ox + cx0, oy + cy0, ox + cx1, oy + cy1
    if ax0 < 0 or ay0 < 0 or ax1 > bw or ay1 > bh:
        return None
    try:
        cloned = cv2.seamlessClone(
            np.ascontiguousarray(patch[cy0:cy1, cx0:cx1]),
            np.ascontiguousarray(base),
            np.ascontiguousarray(hard[cy0:cy1, cx0:cx1]),
            ((ax0 + ax1) // 2, (ay0 + ay1) // 2),
            cv2.NORMAL_CLONE,
        )
    except cv2.error:
        return None
    result = base.copy()
    result[ay0:ay1, ax0:ax1] = cloned[ay0:ay1, ax0:ax1]
    return result


def _blend_alpha(base, patch, alpha, ox, oy):
    result = base.copy()
    bh, bw = base.shape[:2]
    ph, pw = patch.shape[:2]
    bx0, by0 = max(0, ox), max(0, oy)
    bx1, by1 = min(bw, ox + pw), min(bh, oy + ph)
    if bx1 <= bx0 or by1 <= by0:
        return result
    px0, py0 = bx0 - ox, by0 - oy
    px1, py1 = px0 + (bx1 - bx0), py0 + (by1 - by0)
    a = alpha[py0:py1, px0:px1][..., None]
    src = patch[py0:py1, px0:px1].astype(np.float32)
    dst = result[by0:by1, bx0:bx1].astype(np.float32)
    result[by0:by1, bx0:bx1] = np.clip(src * a + dst * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    return result


class Layer:
    """Перенесённое выделение. Состоит из кусков, у каждого свой сдвиг и выравнивание."""

    def __init__(self, image, pieces, feather=12, mode="poisson", faces=0, aligned=0):
        self.image = image
        self.pieces = pieces
        self.feather = feather
        self.mode = mode
        self.color = 1.0
        self.dx = 0.0
        self.dy = 0.0
        self.scale = 1.0
        self.angle = 0.0
        self.visible = True
        self.faces = faces
        self.aligned = aligned
        self.name = "Слой"
        self.erased = []
        self.origins = [self._origin_of(matrix) for mask, matrix in pieces]
        union = np.zeros(image.shape[:2], np.uint8)
        for mask, _ in pieces:
            union |= mask
        ys, xs = np.nonzero(union)
        self.pivot = (float(xs.mean()), float(ys.mean())) if len(xs) else (image.shape[1] / 2.0, image.shape[0] / 2.0)

    @staticmethod
    def _origin_of(matrix):
        return (0.0, 0.0)

    def params(self):
        data = {key: getattr(self, key) for key in ("dx", "dy", "scale", "angle", "feather", "color", "visible", "name")}
        data["erased"] = len(self.erased)
        return data

    def apply_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def _full(self, matrix):
        rot = cv2.getRotationMatrix2D(self.pivot, self.angle, self.scale)
        move = np.array([[1.0, 0.0, self.dx], [0.0, 1.0, self.dy]])
        return _mul(move, _mul(matrix, rot))

    def _warp(self, mask, matrix):
        full = self._full(matrix)
        h, w = mask.shape
        corners = cv2.transform(np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float64), full)[0]
        ox, oy = (int(v) for v in np.floor(corners.min(axis=0)))
        x1, y1 = (int(v) for v in np.ceil(corners.max(axis=0)))
        size = (max(1, x1 - ox), max(1, y1 - oy))
        shifted = full.copy()
        shifted[0, 2] -= ox
        shifted[1, 2] -= oy
        patch = cv2.warpAffine(self.image, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        patch_mask = cv2.warpAffine(mask, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return patch, patch_mask, ox, oy

    @staticmethod
    def _apply_erase(patch_mask, erase, ox, oy):
        ph, pw = patch_mask.shape
        bh, bw = erase.shape
        bx0, by0 = max(0, ox), max(0, oy)
        bx1, by1 = min(bw, ox + pw), min(bh, oy + ph)
        if bx1 <= bx0 or by1 <= by0:
            return patch_mask
        sub = erase[by0:by1, bx0:bx1]
        if not (sub == 0).any():
            return patch_mask
        px0, py0 = bx0 - ox, by0 - oy
        out = patch_mask.copy()
        region = (out[py0:py0 + sub.shape[0], px0:px0 + sub.shape[1]].astype(np.float32) * sub / 255.0)
        out[py0:py0 + sub.shape[0], px0:px0 + sub.shape[1]] = region.astype(np.uint8)
        return out

    def _match_color(self, base, patch, patch_mask, ox, oy):
        bh, bw = base.shape[:2]
        ph, pw = patch.shape[:2]
        bx0, by0 = max(0, ox), max(0, oy)
        bx1, by1 = min(bw, ox + pw), min(bh, oy + ph)
        if bx1 <= bx0 or by1 <= by0:
            return patch
        px0, py0 = bx0 - ox, by0 - oy
        px1, py1 = px0 + (bx1 - bx0), py0 + (by1 - by0)
        matched = transfer(
            patch[py0:py1, px0:px1], base[by0:by1, bx0:bx1], patch_mask[py0:py1, px0:px1], self.color
        )
        out = patch.copy()
        out[py0:py1, px0:px1] = matched
        return out

    def erase(self, x, y, radius):
        """Стирает часть слоя вокруг точки. Координаты — в системе основного снимка."""
        self.erased.append((float(x), float(y), float(radius)))

    def erase_mask(self, shape):
        """Маска стёртого: 0 — стёрто, 255 — оставить."""
        keep = np.full(shape, 255, np.uint8)
        h, w = shape
        for x, y, radius in self.erased:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(keep, (xi, yi), max(1, int(radius)), 0, -1)
        return keep

    def apply(self, base, poisson=True):
        out = base
        used = "alpha"
        erase = self.erase_mask(base.shape[:2]) if self.erased else None
        for mask, matrix in self.pieces:
            patch, patch_mask, ox, oy = self._warp(mask, matrix)
            if erase is not None:
                patch_mask = self._apply_erase(patch_mask, erase, ox, oy)
            if self.color > 0:
                patch = self._match_color(out, patch, patch_mask, ox, oy)
            blended = _blend_poisson(out, patch, patch_mask, ox, oy) if poisson and self.mode == "poisson" else None
            if blended is not None:
                out = blended
                used = "poisson"
            else:
                out = _blend_alpha(out, patch, soft_alpha(patch_mask, self.feather), ox, oy)
        return out, used


def _crop(donor_rgb, selection):
    """Вырезает участок вокруг выделения. Возвращает ещё и сдвиг кропа."""
    box = bbox_of(selection)
    if box is None:
        return None
    h, w = selection.shape
    x0, y0 = max(0, box[0] - CROP_MARGIN), max(0, box[1] - CROP_MARGIN)
    x1, y1 = min(w, box[2] + CROP_MARGIN), min(h, box[3] + CROP_MARGIN)
    crop = np.ascontiguousarray(donor_rgb[y0:y1, x0:x1])
    mask = np.ascontiguousarray(selection[y0:y1, x0:x1])
    origin = np.array([[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)]])
    return crop, mask, origin, (x0, y0)


def build_layer(donor_rgb, selection, donor_faces, base_faces, base_point=None, feather=12, mode="poisson"):
    """Один слой на всё выделение. Если внутри несколько лиц, каждое уходит на своё лицо рабочего снимка."""
    if bbox_of(selection) is None:
        return None
    box = bbox_of(selection)
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    drop = np.array(base_point, np.float64) if base_point is not None else np.array([cx, cy], np.float64)
    offset = drop - np.array([cx, cy], np.float64)

    # Лица ищем по исходному выделению: координаты лиц тоже в системе донора.
    chosen, matches = plan_faces(selection, donor_faces, base_faces, offset)

    if not chosen:
        cropped = _crop(donor_rgb, selection)
        if cropped is None:
            return None
        crop, mask, origin, _ = cropped
        shift = _mul(origin, np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]]]))
        layer = Layer(crop, [(mask, shift)], feather=feather, mode=mode, faces=0, aligned=0)
        layer.origins = [(origin[0, 2], origin[1, 2])]
        return layer

    parts = split_selection(selection, [face[NOSE_TIP] for face in chosen])
    if SMART_EDGES:
        parts = [smart_matte(donor_rgb, part, feather=feather) for part in parts]
    pieces, crop_shifts, aligned = [], [], 0
    crop = mask = origin = None
    for index, (part, face) in enumerate(zip(parts, chosen)):
        cropped = _crop(donor_rgb, part)
        if cropped is None:
            continue
        crop_i, mask_i, origin_i, _ = cropped
        if crop is None:
            crop, origin = crop_i, origin_i
        target_index = matches.get(index)
        matrix = estimate_alignment(face, base_faces[target_index]) if target_index is not None else None
        if matrix is not None:
            aligned += 1
            pieces.append((mask_i, _mul(matrix, origin_i)))
            crop_shifts.append((origin_i[0, 2], origin_i[1, 2]))
        else:
            guess = _mul(origin_i, np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]]]))
            pieces.append((mask_i, guess))
            crop_shifts.append((origin_i[0, 2], origin_i[1, 2]))
    if not pieces:
        return None
    layer = Layer(crop, pieces, feather=feather, mode=mode, faces=len(chosen), aligned=aligned)
    layer.origins = crop_shifts
    return layer
