from typing import Optional, Dict

import numpy as np
from PIL import Image

from app.services.image_cache import load_rgb_array, load_gray_array

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pyzbar.pyzbar import decode as zbar_decode
except ImportError:
    zbar_decode = None


def _detect_qr_opencv(rgb: np.ndarray) -> Optional[Dict]:
    if cv2 is None:
        return None

    try:
        detector = cv2.QRCodeDetector()

        # OpenCV wants BGR for some flows, but QR detector works fine on RGB/gray too.
        data, points, _ = detector.detectAndDecode(rgb)
        if data:
            return {"format": "QR_CODE", "value": data}

        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(rgb)
        if ok and decoded_info:
            first = next((x for x in decoded_info if x), None)
            if first:
                return {"format": "QR_CODE", "value": first}
    except Exception as e:
        print("QR ERROR:", repr(e))

    return None


def _detect_barcode_pyzbar(gray: np.ndarray) -> Optional[Dict]:
    if zbar_decode is None:
        return None

    try:
        decoded = zbar_decode(Image.fromarray(gray))
        if decoded:
            item = decoded[0]
            return {
                "format": getattr(item, "type", "BARCODE"),
                "value": item.data.decode("utf-8", errors="ignore"),
            }
    except Exception as e:
        print("BARCODE ERROR:", repr(e))

    return None


def detect_barcode_or_qr(path: str) -> Optional[Dict]:
    """
    Fast path:
    1) QR via OpenCV
    2) barcode/QR via pyzbar
    """
    rgb = load_rgb_array(path)
    qr_result = _detect_qr_opencv(rgb)
    if qr_result:
        return qr_result

    gray = load_gray_array(path)
    return _detect_barcode_pyzbar(gray)