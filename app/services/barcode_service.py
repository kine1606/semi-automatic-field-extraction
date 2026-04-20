from typing import Any

import cv2


def _detect_qr_with_opencv(image) -> dict[str, str] | None:
    detector = cv2.QRCodeDetector()

    # Try multi-QR first.
    ok, decoded_info, _, _ = detector.detectAndDecodeMulti(image)
    if ok and decoded_info:
        for value in decoded_info:
            if value:
                return {"format": "QR_CODE", "value": value}

    # Fallback to single QR decode.
    value, _, _ = detector.detectAndDecode(image)
    if value:
        return {"format": "QR_CODE", "value": value}

    return None


def _detect_barcode_with_opencv(image) -> dict[str, str] | None:
    # OpenCV barcode API availability depends on build/version.
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        return None

    detector = cv2.barcode_BarcodeDetector()

    # Handle different return signatures across OpenCV versions.
    result = detector.detectAndDecode(image)
    if not isinstance(result, tuple):
        return None

    decoded_values: list[str] = []
    decoded_types: list[Any] = []

    for item in result:
        if isinstance(item, (list, tuple)):
            if item and isinstance(item[0], str):
                decoded_values = [v for v in item if isinstance(v, str)]
            elif item:
                decoded_types = list(item)

    for idx, value in enumerate(decoded_values):
        if value:
            barcode_type = str(decoded_types[idx]) if idx < len(decoded_types) else "BARCODE"
            return {"format": barcode_type, "value": value}

    return None


def detect_barcode_or_qr(image_path: str) -> dict[str, str] | None:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        qr_result = _detect_qr_with_opencv(image)
        if qr_result:
            return qr_result

        barcode_result = _detect_barcode_with_opencv(image)
        if barcode_result:
            return barcode_result

        return None
    except Exception:
        return None
