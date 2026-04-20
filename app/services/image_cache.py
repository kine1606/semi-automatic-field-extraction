from functools import lru_cache
from PIL import Image, ImageOps
import numpy as np

_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
_MAX_SIDE = 1600  # should match your Stage A processed image size


def _resize_if_needed(img: Image.Image, max_side: int = _MAX_SIDE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img

    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, _RESAMPLE)


@lru_cache(maxsize=64)
def load_rgb_array(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = _resize_if_needed(im)
        return np.array(im)


@lru_cache(maxsize=64)
def load_gray_array(path: str) -> np.ndarray:
    rgb = load_rgb_array(path).astype(np.uint8)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
    return gray


def clear_image_cache() -> None:
    load_rgb_array.cache_clear()
    load_gray_array.cache_clear()