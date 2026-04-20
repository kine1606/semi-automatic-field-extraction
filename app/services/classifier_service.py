from rapidfuzz import fuzz
from app.services.text_utils import normalize_text
from app.services.ocr_service import extract_text_variants
from app.services.image_features import estimate_green_ratio
from app.services.barcode_service import detect_barcode_or_qr  # NEW

# --- Keyword definitions ---

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

# --- Fuzzy matching (fixes OCR noise like "nang iuong") ---

def fuzzy_keyword_score(text: str, keywords: list[str], threshold: int = 80) -> int:
    """
    Score how many keywords fuzzy-match inside the text.
    Uses token_set_ratio so word order / extra words don't penalise.
    Returns a raw count (same interface as the old exact counter).
    """
    score = 0
    words = text.split()
    window_size = 4  # check sliding windows of up to 4 words
    windows = [
        " ".join(words[i:i + window_size])
        for i in range(max(1, len(words) - window_size + 1))
    ]
    for kw in keywords:
        if any(fuzz.token_set_ratio(kw, w) >= threshold for w in windows):
            score += 1
    return score


# --- Confidence from signal strength (fixes magic numbers) ---

def energy_confidence(energy_score: int, green_ratio: float) -> float:
    base = min(0.55 + energy_score * 0.12, 0.92)
    if green_ratio > 0.18:
        base = min(base + 0.06, 0.95)
    return round(base, 2)


def nameplate_confidence(nameplate_score: int) -> float:
    return round(min(0.55 + nameplate_score * 0.10, 0.90), 2)


# --- Main classifier ---

def classify_image(processed_path: str) -> dict:
    # Step 1: barcode / QR — fast, no OCR needed
    barcode_result = detect_barcode_or_qr(processed_path)
    if barcode_result:
        return {
            "imageType": "barcode_qr",
            "imageTypeConfidence": 0.97,
            "reason": f"Barcode/QR detected: {barcode_result.get('format')}",
            "ocrText": "",
            "signals": {"barcode": barcode_result},
        }

    # Step 2: OCR — only if no barcode
    ocr_variants = extract_text_variants(processed_path)
    merged_text = " ".join(v for v in ocr_variants.values() if v)
    normalized = normalize_text(merged_text)

    # Guard: did OCR actually run?
    ocr_empty = len(normalized.strip()) == 0

    # Step 3: Score all keyword families
    energy_score   = fuzzy_keyword_score(normalized, ENERGY_KEYWORDS)
    nameplate_score = fuzzy_keyword_score(normalized, NAMEPLATE_KEYWORDS)
    receipt_score  = fuzzy_keyword_score(normalized, RECEIPT_KEYWORDS)
    green_ratio    = estimate_green_ratio(processed_path)
    text_length    = len(normalized)

    # Collect all signals for logging / debugging
    signals = {
        "energy_score": energy_score,
        "nameplate_score": nameplate_score,
        "receipt_score": receipt_score,
        "green_ratio": round(green_ratio, 3),
        "text_length": text_length,
        "ocr_empty": ocr_empty,
    }

    print("OCR VARIANTS:")
    for k, v in ocr_variants.items():
        print(f"[{k}] -> {repr(v[:300])}")

    print("MERGED:", repr(merged_text[:500]))
    print("NORMALIZED:", repr(normalized[:500]))
    print("SIGNALS:", signals)
    # Step 4: Receipt — check before energy/nameplate (distinct keyword set)
    if receipt_score >= 2:
        return {
            "imageType": "receipt_document",
            "imageTypeConfidence": round(min(0.60 + receipt_score * 0.08, 0.91), 2),
            "reason": f"Receipt keywords={receipt_score}",
            "ocrText": merged_text,
            "signals": signals,
        }

    # Step 5: Energy vs nameplate — compare scores, no silent tie-break
    if energy_score > 0 or nameplate_score > 0:
        # Weighted score: green_ratio boosts energy only
        energy_weighted   = energy_score + (1.5 if green_ratio > 0.18 else 0)
        nameplate_weighted = nameplate_score

        if energy_weighted >= nameplate_weighted and energy_score >= 1:
            return {
                "imageType": "energy_label_vn",
                "imageTypeConfidence": energy_confidence(energy_score, green_ratio),
                "reason": (
                    f"Energy keywords={energy_score}, "
                    f"green_ratio={green_ratio:.2f}, "
                    f"nameplate_score={nameplate_score}"
                ),
                "ocrText": merged_text,
                "signals": signals,
            }

        if nameplate_score >= 1:
            return {
                "imageType": "nameplate_label",
                "imageTypeConfidence": nameplate_confidence(nameplate_score),
                "reason": (
                    f"Nameplate keywords={nameplate_score}, "
                    f"energy_score={energy_score}"
                ),
                "ocrText": merged_text,
                "signals": signals,
            }

    # Step 6: Very little text — but distinguish OCR failure from sparse image
    if text_length < 20:
        if ocr_empty:
            return {
                "imageType": "raw_equipment_photo",
                "imageTypeConfidence": 0.55,
                "reason": "OCR returned no text — likely raw equipment photo or blurry image",
                "ocrText": merged_text,
                "signals": signals,
            }
        return {
            "imageType": "raw_equipment_photo",
            "imageTypeConfidence": 0.50,
            "reason": f"Very short text ({text_length} chars), no matching keywords",
            "ocrText": merged_text,
            "signals": signals,
        }

    # Step 7: Genuine unknown — log all signals
    return {
        "imageType": "unknown",
        "imageTypeConfidence": 0.35,
        "reason": "No rule matched — check signals for manual review",
        "ocrText": merged_text,
        "signals": signals,
    }

