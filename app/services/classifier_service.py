import time
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz

from app.services.text_utils import normalize_text
from app.services.ocr_service import extract_text_variants
from app.services.image_features import estimate_green_ratio
from app.services.barcode_service import detect_barcode_or_qr

ENERGY_KEYWORDS = [
    "nang luong", "nhan nang luong", "bo cong thuong",
    "tieu thu dien", "kwh", "hieu suat", "sao", "tcvn",
]

NAMEPLATE_KEYWORDS = [
    "model", "serial", "sn", "220v", "240v", "hz",
    "made in", "voltage", "frequency", "watt", "ampere",
]

RECEIPT_KEYWORDS = [
    "tong cong", "thue vat", "hoa don", "thanh tien",
    "so luong", "don gia", "nha hang", "sieu thi",
]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _build_windows(text: str, max_window_size: int = 4) -> list[str]:
    words = text.split()
    if not words:
        return []

    windows: list[str] = []
    n = len(words)

    for size in range(1, max_window_size + 1):
        if n < size:
            break
        for i in range(n - size + 1):
            windows.append(" ".join(words[i:i + size]))

    # dedupe while preserving order
    return list(dict.fromkeys(windows))


def fuzzy_keyword_score(text: str, keywords: list[str], threshold: int = 80) -> int:
    """
    Restores the old 'look for a keyword inside local OCR windows' behavior,
    which is much more stable than comparing the whole OCR blob to each keyword.
    """
    if not text:
        return 0

    windows = _build_windows(text, max_window_size=4)
    score = 0

    for kw in keywords:
        # fast exact shortcut
        if kw in text:
            score += 1
            continue

        matched = False
        for w in windows:
            if fuzz.token_set_ratio(kw, w) >= threshold:
                matched = True
                break

        if matched:
            score += 1

    return score


def energy_confidence(energy_score: int, green_ratio: float) -> float:
    base = min(0.55 + energy_score * 0.12, 0.92)
    if green_ratio > 0.18:
        base = min(base + 0.06, 0.95)
    return round(base, 2)


def nameplate_confidence(nameplate_score: int) -> float:
    return round(min(0.55 + nameplate_score * 0.10, 0.90), 2)


def _build_result(
    image_type: str,
    confidence: float,
    reason: str,
    merged_text: str,
    signals: dict,
    include_debug: bool,
) -> dict:
    result = {
        "imageType": image_type,
        "imageTypeConfidence": confidence,
        "reason": reason,
    }

    if include_debug:
        result["ocrText"] = merged_text
        result["signals"] = signals

    return result


def _print_debug(path: str, ocr_variants: dict, merged: str, normalized: str, signals: dict) -> None:
    print(f"\n── {path} ──")
    for k, v in ocr_variants.items():
        print(f"  [{k}] {repr(v[:200])}")
    print(f"  merged:     {repr(merged[:300])}")
    print(f"  normalized: {repr(normalized[:300])}")
    print(f"  signals:    {signals}")


def classify_image(path: str, include_debug: bool = False) -> dict:
    """
    Safe, accurate single-image classifier.
    Keeps the old helper contracts (path in, result out).
    """
    t0 = time.perf_counter()

    barcode_result = detect_barcode_or_qr(path)
    if barcode_result:
        signals = {"barcode": barcode_result, "_ms": _ms(t0)}
        return _build_result(
            "barcode_qr",
            0.97,
            f"Barcode/QR detected: {barcode_result.get('format')}",
            "",
            signals,
            include_debug,
        )

    ocr_variants = extract_text_variants(path)
    merged_text = " ".join(v for v in ocr_variants.values() if v)
    normalized = normalize_text(merged_text)

    ocr_empty = not normalized.strip()
    text_length = len(normalized)

    energy_score = fuzzy_keyword_score(normalized, ENERGY_KEYWORDS)
    nameplate_score = fuzzy_keyword_score(normalized, NAMEPLATE_KEYWORDS)
    receipt_score = fuzzy_keyword_score(normalized, RECEIPT_KEYWORDS)
    green_ratio = estimate_green_ratio(path)

    signals = {
        "energy_score": energy_score,
        "nameplate_score": nameplate_score,
        "receipt_score": receipt_score,
        "green_ratio": round(green_ratio, 3),
        "text_length": text_length,
        "ocr_empty": ocr_empty,
        "_ms": _ms(t0),
    }

    if include_debug:
        _print_debug(path, ocr_variants, merged_text, normalized, signals)

    if receipt_score >= 2:
        return _build_result(
            "receipt_document",
            round(min(0.60 + receipt_score * 0.08, 0.91), 2),
            f"Receipt keywords={receipt_score}",
            merged_text,
            signals,
            include_debug,
        )

    if energy_score > 0 or nameplate_score > 0:
        energy_weighted = energy_score + (1.5 if green_ratio > 0.18 else 0)
        nameplate_weighted = nameplate_score

        if energy_weighted >= nameplate_weighted and energy_score >= 1:
            return _build_result(
                "energy_label_vn",
                energy_confidence(energy_score, green_ratio),
                (
                    f"Energy keywords={energy_score}, "
                    f"green_ratio={green_ratio:.2f}, "
                    f"nameplate_score={nameplate_score}"
                ),
                merged_text,
                signals,
                include_debug,
            )

        if nameplate_score >= 1:
            return _build_result(
                "nameplate_label",
                nameplate_confidence(nameplate_score),
                (
                    f"Nameplate keywords={nameplate_score}, "
                    f"energy_score={energy_score}"
                ),
                merged_text,
                signals,
                include_debug,
            )

    # Visual fallback for obvious green labels when OCR is weak
    if ocr_empty and green_ratio > 0.25:
        return _build_result(
            "energy_label_vn",
            0.58,
            "Green-dominant label appearance, but OCR failed",
            merged_text,
            signals,
            include_debug,
        )

    if text_length < 20:
        if ocr_empty:
            return _build_result(
                "raw_equipment_photo",
                0.55,
                "OCR returned no text — likely raw equipment photo or blurry image",
                merged_text,
                signals,
                include_debug,
            )

        return _build_result(
            "raw_equipment_photo",
            0.50,
            f"Very short text ({text_length} chars), no matching keywords",
            merged_text,
            signals,
            include_debug,
        )

    return _build_result(
        "unknown",
        0.35,
        "No rule matched — check debug signals for manual review",
        merged_text,
        signals,
        include_debug,
    )


def classify_images(
    paths: list[str],
    include_debug: bool = False,
    max_workers: int = 3,
) -> list[dict]:
    """
    Fast version without changing classification behavior:
    parallelize across images, not inside one image.
    """
    if not paths:
        return []

    workers = min(max_workers, len(paths))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(classify_image, path, include_debug) for path in paths]
        return [f.result() for f in futures]