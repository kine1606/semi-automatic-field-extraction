import os
from typing import Dict

import pytesseract
from PIL import Image, ImageOps, ImageEnhance

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_LANG = os.getenv("OCR_LANG", "vie+eng")
FAST_CONFIG = os.getenv("OCR_FAST_CONFIG", "--oem 3 --psm 6")
SLOW_CONFIG = os.getenv("OCR_SLOW_CONFIG", "--oem 3 --psm 6")
OCR_TIMEOUT_SEC = int(os.getenv("OCR_TIMEOUT_SEC", "5"))


def _open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


def _resize_for_ocr(img: Image.Image, max_side: int = 1200) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((int(w * scale), int(h * scale)))


def _run_ocr(img: Image.Image, config: str) -> str:
    try:
        return (
            pytesseract.image_to_string(
                img,
                lang=OCR_LANG,
                config=config,
                timeout=OCR_TIMEOUT_SEC,
            )
            or ""
        ).strip()
    except Exception as e:
        print("OCR ERROR:", repr(e))
        return ""


def extract_text_fast(path: str) -> str:
    """
    Stage B OCR: one cheap grayscale pass.
    """
    img = _resize_for_ocr(_open_rgb(path), max_side=1200)
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    return _run_ocr(gray, FAST_CONFIG)


def extract_text_variants(path: str) -> Dict[str, str]:
    """
    Stage C / fallback OCR: only call this when needed.
    """
    img = _resize_for_ocr(_open_rgb(path), max_side=1400)

    rgb = img
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    bw = gray.point(lambda p: 255 if p > 160 else 0)

    return {
        "rgb": _run_ocr(rgb, SLOW_CONFIG),
        "gray": _run_ocr(gray, SLOW_CONFIG),
        "bw": _run_ocr(bw, SLOW_CONFIG),
    }