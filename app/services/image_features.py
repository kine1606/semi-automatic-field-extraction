import numpy as np
from PIL import Image, ImageOps


def estimate_green_ratio(path: str) -> float:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((160, 160))

    arr = np.array(img).astype(np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_like = (g > r + 20) & (g > b + 20) & (g > 70)
    return float(green_like.mean())