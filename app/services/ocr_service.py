from PIL import Image, ImageOps
import pytesseract


def _ocr_image(img: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(img, lang="vie+eng")
        return text or ""
    except Exception as e:
        print("OCR ERROR:", repr(e))
        return ""


def extract_text(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            return _ocr_image(img)
    except Exception:
        return ""


def extract_text_variants(image_path: str) -> dict[str, str]:
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            gray = ImageOps.grayscale(rgb)
            bw = gray.point(lambda p: 255 if p > 150 else 0)

            return {
                "rgb": _ocr_image(rgb),
                "gray": _ocr_image(gray),
                "bw": _ocr_image(bw),
            }
    except Exception:
        return {"rgb": "", "gray": "", "bw": ""}
