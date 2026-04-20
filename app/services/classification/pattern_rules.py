import re


def collect_nameplate_pattern_score(text: str) -> tuple[int, dict]:
    patterns = {
        "voltage": r"\b(110|120|220|230|240|380|400|415)\s*v\b",
        "voltage_range": r"\b\d{3}\s*-\s*\d{3}\s*v\b",
        "hz": r"\b(50|60)\s*hz\b",
        "phase": r"\b\d+\s*ph\b|\b\d+\s*/\s*\d+\s*/\s*\d+\b",
        "amp": r"\b\d+(\.\d+)?\s*a\b",
        "watt": r"\b\d+(\.\d+)?\s*w\b",
        "kg": r"\b\d+(\.\d+)?\s*kg\b",
        "liter": r"\b\d+(\.\d+)?\s*l\b",
        "btu": r"\b\d+(\.\d+)?\s*btu\b",
        "pressure_bar": r"\b\d+(\.\d+)?\s*bar\b",
        "pressure_mpa": r"\b\d+(\.\d+)?\s*mpa\b",
        "refrigerant": r"\br\d{2,4}[a-z]?\b",
        "ip_rating": r"\bipx?\d+\b",
        "model_code": r"\b[a-z]{1,6}[-/][a-z0-9-]{2,}\b",
        "serial_phrase": r"\bso\s*seri\b|\bserial\b|\bs\/n\b|\bs\s*no\b",
        "year": r"\b(20\d{2}|19\d{2})\b",
    }

    matched = {}
    score = 0

    for name, pattern in patterns.items():
        hit = re.search(pattern, text, flags=re.IGNORECASE)
        if hit:
            score += 1
            matched[name] = hit.group(0)

    return score, matched


__all__ = ["collect_nameplate_pattern_score"]
