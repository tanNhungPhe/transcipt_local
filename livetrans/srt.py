"""SubRip (.srt) formatting."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{i}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text.strip()}\n"
        for i, c in enumerate((c for c in cues if c.text.strip()), start=1)
    ]
    return "\n".join(blocks)
