import re
import time
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from app.services.text_utils import normalize_text
from app.services.ocr_service import extract_text_fast, extract_text_variants
from app.services.image_features import estimate_green_ratio
from app.services.barcode_service import detect_barcode_or_qr


ENERGY_STRONG = [
    "nhan nang luong",
    "nang luong",
    "bo cong thuong",
    "tieu thu dien",
    "tieu chuan viet nam",
    "chi so hsnl",
]

ENERGY_WEAK = [
    "kwh",
    "tcvn",
    "sao",
]

NAMEPLATE_STRONG = [
    "model",
    "serial",
    "made in",
    "voltage",
    "frequency",
]

NAMEPLATE_WEAK = [
    "220v",
    "240v",
    "hz",
    "watt",
    "ampere",
]

RECEIPT_STRONG = [
    "hoa don",
    "tong cong",
    "thue vat",
    "thanh tien",
    "so luong",
    "don gia",
]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _build_result(image_type, confidence, reason, merged_text, signals, include_debug):
    result = {
        "imageType": image_type,
        "imageTypeConfidence": confidence,
        "reason": reason,
    }
    if include_debug:
        result["ocrText"] = merged_text
        result["signals"] = signals
    return result


def _contains_exact(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _windows_for_phrase(text: str, phrase: str) -> list[str]:
    words = text.split()
    n = len(phrase.split())
    if not words:
        return []

    sizes = sorted({max(1, n - 1), n, n + 1})
    windows = []

    for size in sizes:
        if len(words) < size:
            continue
        for i in range(len(words) - size + 1):
            windows.append(" ".join(words[i:i + size]))

    return windows


def _fuzzy_phrase_hit(text: str, phrase: str, threshold: int = 90) -> bool:
    if len(phrase.split()) < 2:
        return False

    for window in _windows_for_phrase(text, phrase):
        if fuzz.token_sort_ratio(phrase, window) >= threshold:
            return True
    return False


def _score_strong_phrases(text: str, phrases: list[str]) -> tuple[int, list[str]]:
    hits = []
    for phrase in phrases:
        if _contains_exact(text, phrase) or _fuzzy_phrase_hit(text, phrase, threshold=90):
            hits.append(phrase)
    return len(hits), hits


def _score_weak_keywords(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    hits = []
    for kw in keywords:
        if _contains_exact(text, kw):
            hits.append(kw)
    return len(hits), hits


def energy_confidence(energy_strong: int, energy_weak: int, green_ratio: float) -> float:
    base = 0.70 + min(energy_strong, 3) * 0.08 + min(energy_weak, 2) * 0.03
    if green_ratio > 0.18:
        base += 0.05
    return round(min(base, 0.97), 2)


def nameplate_confidence(nameplate_strong: int, nameplate_weak: int) -> float:
    base = 0.68 + min(nameplate_strong, 3) * 0.08 + min(nameplate_weak, 2) * 0.03
    return round(min(base, 0.93), 2)


def receipt_confidence(receipt_strong: int) -> float:
    base = 0.70 + min(receipt_strong, 3) * 0.07
    return round(min(base, 0.91), 2)


def _classify_from_text(
    normalized: str,
    green_ratio: float,
    merged_text: str,
    include_debug: bool,
    t0: float,
):
    ocr_empty = not normalized.strip()
    text_length = len(normalized)

    energy_strong, energy_strong_hits = _score_strong_phrases(normalized, ENERGY_STRONG)
    energy_weak, energy_weak_hits = _score_weak_keywords(normalized, ENERGY_WEAK)

    nameplate_strong, nameplate_strong_hits = _score_strong_phrases(normalized, NAMEPLATE_STRONG)
    nameplate_weak, nameplate_weak_hits = _score_weak_keywords(normalized, NAMEPLATE_WEAK)

    receipt_strong, receipt_hits = _score_strong_phrases(normalized, RECEIPT_STRONG)

    signals = {
        "energy_strong": energy_strong,
        "energy_weak": energy_weak,
        "nameplate_strong": nameplate_strong,
        "nameplate_weak": nameplate_weak,
        "receipt_strong": receipt_strong,
        "green_ratio": round(green_ratio, 3),
        "text_length": text_length,
        "ocr_empty": ocr_empty,
        "_ms": _ms(t0),
    }

    if include_debug:
        signals["matched"] = {
            "energy_strong": energy_strong_hits,
            "energy_weak": energy_weak_hits,
            "nameplate_strong": nameplate_strong_hits,
            "nameplate_weak": nameplate_weak_hits,
            "receipt_strong": receipt_hits,
        }

    if receipt_strong >= 2 and green_ratio < 0.12 and energy_strong == 0:
        return _build_result(
            "receipt_document",
            receipt_confidence(receipt_strong),
            f"Receipt phrases={receipt_strong}",
            merged_text,
            signals,
            include_debug,
        )

    is_energy = (
        energy_strong >= 2
        or (green_ratio > 0.18 and (energy_strong >= 1 or energy_weak >= 2))
        or ("nhan nang luong" in normalized and green_ratio > 0.12)
    )

    if is_energy:
        return _build_result(
            "energy_label_vn",
            energy_confidence(energy_strong, energy_weak, green_ratio),
            (
                f"Energy strong={energy_strong}, "
                f"energy weak={energy_weak}, "
                f"green_ratio={green_ratio:.2f}"
            ),
            merged_text,
            signals,
            include_debug,
        )

    is_nameplate = (
        nameplate_strong >= 2
        or (
            nameplate_strong >= 1
            and nameplate_weak >= 1
            and energy_strong == 0
            and green_ratio < 0.15
        )
    )

    if is_nameplate:
        return _build_result(
            "nameplate_label",
            nameplate_confidence(nameplate_strong, nameplate_weak),
            (
                f"Nameplate strong={nameplate_strong}, "
                f"nameplate weak={nameplate_weak}"
            ),
            merged_text,
            signals,
            include_debug,
        )

    if ocr_empty:
        return _build_result(
            "raw_equipment_photo",
            0.55,
            "OCR returned no text — likely raw equipment photo or blurry image",
            merged_text,
            signals,
            include_debug,
        )

    if text_length < 80 and energy_strong == 0 and nameplate_strong == 0 and receipt_strong == 0:
        return _build_result(
            "raw_equipment_photo",
            0.58,
            f"Short OCR text ({text_length} chars) with no strong matches",
            merged_text,
            signals,
            include_debug,
        )

    return _build_result(
        "unknown",
        0.35,
        "No strong rule matched — manual review recommended",
        merged_text,
        signals,
        include_debug,
    )


def classify_image(path: str, include_debug: bool = False) -> dict:
    t0 = time.perf_counter()

    # cheap features first
    green_ratio = estimate_green_ratio(path)

    barcode_result = detect_barcode_or_qr(path)
    if barcode_result:
        signals = {"barcode": barcode_result, "green_ratio": round(green_ratio, 3), "_ms": _ms(t0)}
        return _build_result(
            "barcode_qr",
            0.97,
            f"Barcode/QR detected: {barcode_result.get('format')}",
            "",
            signals,
            include_debug,
        )

    # Stage B fast OCR: one pass first
    fast_text = extract_text_fast(path)
    merged_text = fast_text.strip()
    normalized = normalize_text(merged_text)

    # Fast happy path for obvious green energy labels
    if green_ratio > 0.18:
        quick_result = _classify_from_text(normalized, green_ratio, merged_text, include_debug, t0)
        if quick_result["imageType"] == "energy_label_vn":
            return quick_result

    # For obvious non-label images, avoid heavy OCR fallback
    if green_ratio < 0.05:
        quick_result = _classify_from_text(normalized, green_ratio, merged_text, include_debug, t0)
        if quick_result["imageType"] in {"raw_equipment_photo", "unknown"}:
            return quick_result

    # Only now do heavier OCR fallback
    variants = extract_text_variants(path)
    deduped = []
    seen = set()
    for value in variants.values():
        value = (value or "").strip()
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)

    merged_text = " ".join(deduped)
    normalized = normalize_text(merged_text)

    return _classify_from_text(normalized, green_ratio, merged_text, include_debug, t0)


def classify_images(
    paths: list[str],
    include_debug: bool = False,
    max_workers: int = 3,
) -> list[dict]:
    if not paths:
        return []

    workers = min(max_workers, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(classify_image, p, include_debug) for p in paths]
        return [f.result() for f in futures]