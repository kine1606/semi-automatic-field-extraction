import re
from typing import Optional, Tuple

from app.services.text_utils import normalize_text

KNOWN_BRANDS = [
    "DAIKIN",
    "MITSUBISHI ELECTRIC",
    "FUNIKI",
    "LG",
    "SAMSUNG",
    "PANASONIC",
    "BOSCH",
    "TOSHIBA",
    "ELECTROLUX",
    "SHARP",
    "AQUA",
    "HITACHI",
    "MIDEA",
    "CASPER",
]

MODEL_CODE_CAPTURE_RE = r"\b((?=[A-Z0-9/\-]{5,}\b)(?=[A-Z0-9/\-]*\d)[A-Z][A-Z0-9/\-]+)\b"


def clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip(" \t:;,.|-")
    return value or None


def make_candidate(
    value: str | None,
    confidence: float,
    source: str,
    evidence: list[str] | None = None,
) -> dict | None:
    value = clean_value(value)
    if not value:
        return None

    return {
        "value": value,
        "confidence": round(confidence, 2),
        "source": source,
        "evidence": evidence or [],
    }


def build_lines(raw_text: str) -> list[dict]:
    lines = []
    seen = set()

    for raw in raw_text.splitlines():
        raw = clean_value(raw)
        if not raw or len(raw) < 2:
            continue

        norm = normalize_text(raw)
        if not norm or len(norm) < 2:
            continue

        if norm in seen:
            continue

        seen.add(norm)
        lines.append({
            "raw": raw,
            "norm": norm,
        })

    return lines


def merge_ocr_variants(ocr_variants: dict) -> str:
    merged_lines = []
    seen = set()

    for text in ocr_variants.values():
        for line in build_lines(text or ""):
            if line["norm"] in seen:
                continue
            seen.add(line["norm"])
            merged_lines.append(line["raw"])

    return "\n".join(merged_lines)


def search_patterns(text: str, patterns: list[str]) -> tuple[str | None, list[str]]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip() if match.groups() else match.group(0).strip()
            return clean_value(value), [clean_value(match.group(0))]
    return None, []


def extract_after_labels(
    lines: list[dict],
    labels: list[str],
    value_patterns: list[str] | None = None,
    lookahead: int = 1,
) -> tuple[str | None, list[str]]:
    value_patterns = value_patterns or []

    for idx, line in enumerate(lines):
        if not any(label in line["norm"] for label in labels):
            continue

        raw_line = line["raw"]

        # Same line, after ":" or ";"
        parts = re.split(r"[:;]", raw_line, maxsplit=1)
        if len(parts) == 2:
            rhs = clean_value(parts[1])
            if rhs:
                if value_patterns:
                    value, _ = search_patterns(rhs, value_patterns)
                    if value:
                        return value, [raw_line]
                else:
                    return rhs, [raw_line]

        # Same line, no colon, extract by pattern
        if value_patterns:
            value, _ = search_patterns(raw_line, value_patterns)
            if value:
                return value, [raw_line]

        # Next line(s)
        for step in range(1, lookahead + 1):
            j = idx + step
            if j >= len(lines):
                break

            candidate_line = clean_value(lines[j]["raw"])
            if not candidate_line:
                continue

            if value_patterns:
                value, _ = search_patterns(candidate_line, value_patterns)
                if value:
                    return value, [raw_line, candidate_line]
            else:
                return candidate_line, [raw_line, candidate_line]

    return None, []


def find_known_brand(raw_text: str) -> tuple[str | None, list[str]]:
    upper = raw_text.upper()

    for brand in KNOWN_BRANDS:
        if brand in upper:
            return brand, [brand]

    lines = build_lines(raw_text)
    for line in lines[:4]:
        raw = line["raw"].strip()
        # fallback: big uppercase brand-like heading
        if re.fullmatch(r"[A-Z][A-Z0-9& .\-]{3,}", raw.upper()):
            return raw.upper(), [raw]

    return None, []


def infer_category(raw_text: str) -> tuple[str | None, list[str]]:
    norm = normalize_text(raw_text)

    aircon_hits = [
        "dieu hoa khong khi",
        "air conditioner",
        "indoor unit",
        "outdoor unit",
        "cooling",
        "btu",
    ]
    fridge_hits = [
        "tu lanh",
        "refrigerator",
        "freezer",
        "food storage",
        "defrost",
    ]
    washer_hits = [
        "may giat",
        "washing machine",
        "spin",
    ]

    if any(hit in norm for hit in aircon_hits) or any(x in norm for x in ["r410a", "r32", "r22"]):
        return "air_conditioner", ["category inferred from AC keywords"]

    if any(hit in norm for hit in fridge_hits) or "r600a" in norm:
        return "refrigerator", ["category inferred from refrigerator keywords"]

    if any(hit in norm for hit in washer_hits):
        return "washing_machine", ["category inferred from washer keywords"]

    return None, []