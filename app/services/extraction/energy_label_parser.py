from app.services.extraction.common import (
    MODEL_CODE_CAPTURE_RE,
    build_lines,
    extract_after_labels,
    find_known_brand,
    infer_category,
    make_candidate,
    search_patterns,
)

SOURCE = "energy_label_ocr"


def _field_from_label_or_pattern(
    lines: list[dict],
    raw_text: str,
    labels: list[str],
    patterns: list[str] | None,
    label_conf: float,
    pattern_conf: float,
):
    value, evidence = extract_after_labels(lines, labels, patterns, lookahead=1)
    if value:
        return make_candidate(value, label_conf, SOURCE, evidence)

    if patterns:
        value, evidence = search_patterns(raw_text, patterns)
        if value:
            return make_candidate(value, pattern_conf, SOURCE, evidence)

    return None


def parse_energy_label(raw_text: str) -> dict:
    lines = build_lines(raw_text)
    fields = {}

    brand, brand_evidence = find_known_brand(raw_text)
    if brand:
        fields["brandName"] = make_candidate(brand, 0.95, SOURCE, brand_evidence)

    model = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["ma san pham", "model"],
        patterns=[MODEL_CODE_CAPTURE_RE],
        label_conf=0.94,
        pattern_conf=0.82,
    )
    if model:
        fields["modelNumber"] = model

    origin = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["xuat xu", "origin", "made in"],
        patterns=None,
        label_conf=0.86,
        pattern_conf=0.78,
    )
    if origin:
        fields["origin"] = origin

    capacity = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["dung tich"],
        patterns=[r"([0-9]+(?:[.,][0-9]+)?\s*[lL])"],
        label_conf=0.90,
        pattern_conf=0.82,
    )
    if capacity:
        fields["capacity"] = capacity

    annual_energy = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["dien nang tieu thu", "tieu thu dien"],
        patterns=[r"([0-9]+(?:[.,][0-9]+)?\s*kwh\s*/?\s*(?:nam|year)?)"],
        label_conf=0.92,
        pattern_conf=0.84,
    )
    if annual_energy:
        fields["annualEnergyConsumption"] = annual_energy

    standard = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["tieu chuan viet nam"],
        patterns=[r"(TCVN\s*[0-9:.\-/]+)"],
        label_conf=0.90,
        pattern_conf=0.82,
    )
    if standard:
        fields["standard"] = standard

    efficiency_index = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["chi so hsnl"],
        patterns=[r"([0-9]+(?:[.,][0-9]+)?)"],
        label_conf=0.86,
        pattern_conf=0.78,
    )
    if efficiency_index:
        fields["efficiencyIndex"] = efficiency_index

    cert_no = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["so chung nhan", "so chong nhan"],
        patterns=[r"(?:No\.?\s*)?([A-Z0-9\-]+)"],
        label_conf=0.88,
        pattern_conf=0.80,
    )
    if cert_no:
        fields["certificateNumber"] = cert_no

    category, category_evidence = infer_category(raw_text)
    if category:
        fields["categoryType"] = make_candidate(category, 0.68, SOURCE, category_evidence)

    return fields