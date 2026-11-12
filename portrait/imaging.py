import numpy as np
from PIL import Image, ImageOps

JPEG_QUALITY = 95


def load_rgb(path):
    with Image.open(path) as img:
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def save_jpeg(rgb, path, exif=None):
    kwargs = {"quality": JPEG_QUALITY, "subsampling": 0}
    if exif:
        kwargs["exif"] = exif
    Image.fromarray(rgb).save(path, "JPEG", **kwargs)


def make_thumbnail(src, dst, size=256):
    with Image.open(src) as img:
        img.draft("RGB", (size * 2, size * 2))
        thumb = ImageOps.exif_transpose(img).convert("RGB")
        thumb.thumbnail((size, size))
        thumb.save(dst, "JPEG", quality=85)
