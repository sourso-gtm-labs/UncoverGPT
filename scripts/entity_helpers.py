"""Pure entity-normalization helpers used by the run-level experiment."""

from __future__ import annotations

import re
from functools import lru_cache


LLM_SCORE_THRESHOLD = 0.5
KEEP_LABELS = {"ORG", "PERSON", "PRODUCT", "GPE", "NORP", "WORK_OF_ART", "EVENT"}

_ALIAS = {
    "tsla": "tesla",
    "tesla inc": "tesla",
    "tesla inc.": "tesla",
    "tesla, inc": "tesla",
    "elon musk's": "elon musk",
    "musk": "elon musk",
    "donald trump's": "donald trump",
    "u s": "united states",
    "u.s": "united states",
    "us": "united states",
    "meta platforms": "meta",
    "meta platforms inc": "meta",
    "meta inc": "meta",
    "facebook": "meta",
    "meta superintelligence labs": "meta",
    "alphabet": "google",
    "alphabet inc": "google",
    "google llc": "google",
    "microsoft corp": "microsoft",
    "microsoft corporation": "microsoft",
    "microsoft research": "microsoft",
    "apple inc": "apple",
    "apple inc.": "apple",
    "salesforce inc": "salesforce",
    "salesforce inc.": "salesforce",
    "crm": "salesforce",
    "nvda": "nvidia",
    "nvidia corp": "nvidia",
    "nvidia corporation": "nvidia",
    "jpm": "jp morgan",
    "jpmorgan": "jp morgan",
    "jpmorgan chase": "jp morgan",
    "fsd": "full self driving",
    "eu": "european union",
    "salesforce ap": "salesforce",
    "salesforce ca": "salesforce",
    "claude opus 4": "claude opus",
    "claude opus 45": "claude opus",
    "claude sonnet 4": "claude sonnet",
    "claude sonnet 45": "claude sonnet",
    "claude haiku 4": "claude haiku",
    "claude haiku 45": "claude haiku",
}

_STOP = {
    "a", "an", "the", "for", "of", "in", "on", "at", "by", "to", "with",
    "from", "and", "or", "but", "is", "are", "was", "be", "it", "its", "as",
}

_GENERIC_TAIL = {
    "stock", "stocks", "price", "prices", "sales", "revenue", "demand",
    "market", "share", "shares", "data", "report", "news", "update", "growth",
    "loss", "profit", "earnings", "filing", "fund", "funds", "sector", "index",
    "rate", "rates", "cost", "costs", "deal", "deals", "flag", "flags", "map",
    "logo", "dictionary", "glossary", "definition", "meaning",
}


def normalize(raw: str) -> str:
    """Normalize an extracted entity string using the experiment aliases."""

    text = raw.strip()
    text = re.sub(r"[''\u2019]s\b", " ", text)
    text = re.sub(r"[:\-/\\]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = _ALIAS.get(text, text)
    words = text.split()
    if words:
        last = words[-1]
        if last.endswith("s") and not last.endswith("ss") and len(last) > 3:
            stem = last[:-1]
            canonical = set(_ALIAS.values())
            full_stem = " ".join(words[:-1] + [stem]) if len(words) > 1 else stem
            if stem in canonical or full_stem in canonical:
                text = full_stem
    return text


def has_chinese(text: str) -> bool:
    """Return whether text contains a CJK Unified Ideograph."""

    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


@lru_cache(maxsize=1)
def load_spacy():
    """Load the English spaCy model once."""

    import spacy

    return spacy.load("en_core_web_sm")


def is_valid_en(raw: str, normalized: str) -> bool:
    """Apply the English entity-candidate filters used in the experiment."""

    if not any(word[0].isupper() for word in raw.split() if word):
        return False
    if len(normalized) < 2 or normalized.replace(" ", "").isdigit():
        return False
    words = normalized.split()
    if any(len(word) == 1 for word in words) or words[0] in _STOP:
        return False
    if re.match(r"^[a-z]{3,}(com|io|org|net)$", normalized):
        return False
    return not (len(words) > 1 and words[-1] in _GENERIC_TAIL)


def merge_aliases(entities: set[str]) -> set[str]:
    """Merge nested aliases using the experiment's longest-first rule."""

    sorted_entities = sorted(entities, key=len, reverse=True)
    merged: list[str] = []
    absorbed: set[str] = set()
    for entity in sorted_entities:
        if entity in absorbed:
            continue
        merged.append(entity)
        for other in sorted_entities:
            if other == entity or other in absorbed:
                continue
            if len(other) >= 4 and (other in entity or entity in other):
                absorbed.add(other)
    return set(merged)
