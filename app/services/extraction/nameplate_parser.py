from app.services.extraction.common import (
    MODEL_CODE_CAPTURE_RE,
    build_lines,
    extract_after_labels,
    find_known_brand,
    infer_category,
    make_candidate,
    search_patterns,
)

SOURCE = "nameplate_ocr"


def _field_from_label_or_pattern(
    lines: list[dict],
    raw_text: str,
    labels: list[str],
    patterns: list[str] | None,
    label_conf: float,
    pattern_conf: float,
    lookahead: int = 1,
):
    value, evidence = extract_after_labels(lines, labels, patterns, lookahead=lookahead)
    if value:
        return make_candidate(value, label_conf, SOURCE, evidence)

    if patterns:
        value, evidence = search_patterns(raw_text, patterns)
        if value:
            return make_candidate(value, pattern_conf, SOURCE, evidence)

    return None


def parse_nameplate(raw_text: str) -> dict:
    lines = build_lines(raw_text)
    fields = {}

    brand, brand_evidence = find_known_brand(raw_text)
    if brand:
        fields["brandName"] = make_candidate(brand, 0.95, SOURCE, brand_evidence)

    model = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["model no", "model n", "model", "ma san pham"],
        patterns=[
            r"(?:model(?:\s*(?:no|n[°o]?))?)\s*[:.]?\s*([A-Z0-9/\-]{4,})",
            MODEL_CODE_CAPTURE_RE,
        ],
        label_conf=0.93,
        pattern_conf=0.82,
    )
    if model:
        fields["modelNumber"] = model

    serial = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["s/no", "s no", "s n", "serial", "so seri", "b/no", "b no"],
        patterns=[
            r"(?:s\/?no|serial|so\s*seri|b\/?no)\s*[:.]?\s*([A-Z0-9\-]{4,})",
        ],
        label_conf=0.92,
        pattern_conf=0.84,
    )
    if serial:
        fields["serialNumber"] = serial

    power_supply = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["rated power supply", "power supply", "dien ap"],
        patterns=[
            r"([0-9]{3}\s*-\s*[0-9]{3}\s*\/\s*[0-9A-Z~]+\s*\/\s*[0-9]{2})",
            r"((?:AC\s*)?[0-9]{2,4}\s*-\s*[0-9]{2,4}\s*V)",
            r"((?:AC\s*)?[0-9]{2,4}\s*V)",
        ],
        label_conf=0.88,
        pattern_conf=0.80,
        lookahead=1,
    )
    if power_supply:
        fields["powerSupply"] = power_supply

    frequency = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["tan so", "frequency", "v/ph/hz"],
        patterns=[r"(\b(?:50|60)\s*Hz\b)"],
        label_conf=0.88,
        pattern_conf=0.82,
    )
    if frequency:
        fields["frequency"] = frequency

    current = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["rated current", "max current", "dong dien", "total rated current"],
        patterns=[r"(\b[0-9]+(?:[.,][0-9]+)?\s*A\b)"],
        label_conf=0.86,
        pattern_conf=0.78,
    )
    if current:
        fields["current"] = current

    rated_power = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["rated power", "total rated power", "cong suat dien", "cong suat"],
        patterns=[r"(\b[0-9]+(?:[.,][0-9]+)?\s*W\b)"],
        label_conf=0.86,
        pattern_conf=0.78,
    )
    if rated_power:
        fields["ratedPower"] = rated_power

    refrigerant = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["refrigerant", "moi chat"],
        patterns=[r"(\bR[0-9]{2,4}[A-Z]?\b)"],
        label_conf=0.92,
        pattern_conf=0.85,
    )
    if refrigerant:
        fields["refrigerant"] = refrigerant

    mass = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["mass", "trong luong"],
        patterns=[r"(\b[0-9]+(?:[.,][0-9]+)?\s*kg\b)"],
        label_conf=0.85,
        pattern_conf=0.80,
    )
    if mass:
        fields["mass"] = mass

    capacity = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["tong dung tich", "dung tich"],
        patterns=[r"(\b[0-9]+(?:[.,][0-9]+)?\s*[lL]\b)"],
        label_conf=0.84,
        pattern_conf=0.78,
    )
    if capacity:
        fields["capacity"] = capacity

    protection = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["protection"],
        patterns=[r"(\bIPX?[0-9]+\b)"],
        label_conf=0.84,
        pattern_conf=0.80,
    )
    if protection:
        fields["protection"] = protection

    year = _field_from_label_or_pattern(
        lines,
        raw_text,
        labels=["year of manufacture"],
        patterns=[r"\b((?:19|20)\d{2})\b"],
        label_conf=0.84,
        pattern_conf=0.76,
    )
    if year:
        fields["yearOfManufacture"] = year

    category, category_evidence = infer_category(raw_text)
    if category:
        fields["categoryType"] = make_candidate(category, 0.78, SOURCE, category_evidence)

    return fields