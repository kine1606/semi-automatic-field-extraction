import numpy as np
from app.services.image_cache import load_rgb_array


def estimate_green_ratio(path: str) -> float:
    """
    Fast, vectorized green-ratio estimate.
    Samples the image instead of scanning every pixel.
    """
    rgb = load_rgb_array(path)

    # downsample cheaply by slicing
    h, w = rgb.shape[:2]
    step_y = max(1, h // 200)
    step_x = max(1, w // 200)
    sample = rgb[::step_y, ::step_x]

    r = sample[:, :, 0].astype(np.int16)
    g = sample[:, :, 1].astype(np.int16)
    b = sample[:, :, 2].astype(np.int16)

    green_like = (g > r + 20) & (g > b + 20) & (g > 70)

    return float(green_like.mean())