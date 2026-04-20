import re

from rapidfuzz import fuzz


def contains_exact(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def score_strong_phrases(text: str, phrases: list[str]) -> tuple[int, list[str]]:
    hits = []
    for phrase in phrases:
        if contains_exact(text, phrase) or fuzzy_phrase_hit(text, phrase, threshold=90):
            hits.append(phrase)
    return len(hits), hits


def score_weak_keywords(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    hits = []
    for kw in keywords:
        if contains_exact(text, kw):
            hits.append(kw)
    return len(hits), hits


def fuzzy_phrase_hit(text: str, phrase: str, threshold: int = 90) -> bool:
    if len(phrase.split()) < 2:
        return False

    for window in windows_for_phrase(text, phrase):
        if fuzz.token_sort_ratio(phrase, window) >= threshold:
            return True
    return False


def windows_for_phrase(text: str, phrase: str) -> list[str]:
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


__all__ = [
    "contains_exact",
    "fuzzy_phrase_hit",
    "score_strong_phrases",
    "score_weak_keywords",
    "windows_for_phrase",
]