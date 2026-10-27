"""Filters for Whisper's well-known hallucinations on silence/music."""
import re

_HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "hãy subscribe",
    "đăng ký kênh",
    "cảm ơn các bạn đã theo dõi",
    "ご視聴ありがとうございました",
    "字幕由",
    "请不吝点赞",
    "amara.org",
)
_ONLY_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_hallucination(text: str) -> bool:
    cleaned = normalize_text(text).lower()
    if not cleaned or _ONLY_PUNCT.match(cleaned):
        return True
    return len(cleaned) < 60 and any(phrase in cleaned for phrase in _HALLUCINATIONS)


def collapse_repeats(text: str, max_repeats: int = 3) -> str:
    """'ha ha ha ha ha ha' -> 'ha ha ha' (Whisper loops on partial audio)."""
    result: list[str] = []
    for word in normalize_text(text).split(" "):
        if len(result) >= max_repeats and all(w == word for w in result[-max_repeats:]):
            continue
        result.append(word)
    return " ".join(result)
