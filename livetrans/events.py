"""Messages passed from the engine thread to the UI thread."""
from dataclasses import dataclass

SPEAKER_ME = "me"


@dataclass(frozen=True)
class UiEvent:
    kind: str                 # status | error | partial | final | translation | speaker | file_done
    utterance_id: int = -1
    text: str = ""
    final: bool = False
    start_sec: float = 0.0
    latency_ms: int = 0
    speaker: str = ""         # "me", "S1", "S2"… ; "" = unknown / not yet resolved
    speaker_pending: bool = False
