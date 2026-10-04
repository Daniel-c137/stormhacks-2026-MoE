"""Language codes as the transcript carries them: ISO 639-1, lowercase (#106)."""

import re

# ISO 639-2/3 codes Scribe or a model may give for common languages, to ISO 639-1.
THREE_TO_TWO = {
    "ara": "ar", "chi": "zh", "zho": "zh", "deu": "de", "ger": "de", "eng": "en", "fas": "fa",
    "per": "fa", "fra": "fr", "fre": "fr", "hin": "hi", "ita": "it", "jpn": "ja", "kor": "ko",
    "nld": "nl", "dut": "nl", "pol": "pl", "por": "pt", "rus": "ru", "spa": "es", "tur": "tr",
    "ukr": "uk", "vie": "vi",
}  # fmt: skip
ISO_CODE = re.compile(r"^[a-z]{2,3}$")


def normalise_language(code: str | None) -> str | None:
    """'ES', ' fr ', 'zh-CN' or 'spa' -> an ISO 639-1 code; None when it isn't a code at all."""
    if not code:
        return None
    base = re.split(r"[-_]", code.strip().lower())[0]
    base = THREE_TO_TWO.get(base, base)
    return base if ISO_CODE.match(base) else None


def is_english(code: str | None) -> bool:
    return normalise_language(code) == "en"
