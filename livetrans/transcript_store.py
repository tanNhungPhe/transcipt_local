"""UI-side transcript model: merges engine events into entries and logs them to disk."""
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entry:
    utterance_id: int
    original: str = ""
    translation: str = ""
    final: bool = False
    translated_final: bool = False
    clock: str = ""


class TranscriptStore:
    def __init__(self, log_dir: Path | None, translate: bool):
        self._translate = translate
        self._entries: dict[int, Entry] = {}
        self._written: set[int] = set()
        self.path: Path | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.path = log_dir / f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt"

    def entries(self) -> list[Entry]:
        return list(self._entries.values())

    def apply(self, kind: str, utterance_id: int, text: str, final: bool) -> bool:
        """Returns True when the visible transcript changed."""
        current = self._entries.get(utterance_id, Entry(utterance_id, clock=f"{datetime.now():%H:%M:%S}"))
        if kind == "partial":
            if current.final:
                return False
            updated = replace(current, original=text)
        elif kind == "final":
            if not text:
                self._entries.pop(utterance_id, None)
                return True
            updated = replace(current, original=text, final=True)
        elif kind == "translation":
            if current.translated_final and not final:
                return False  # a late partial translation must not overwrite the final one
            updated = replace(current, translation=text, translated_final=final)
        else:
            return False
        self._entries[utterance_id] = updated
        if self._is_complete(updated):
            self._write(updated)
        return True

    def flush(self) -> None:
        for entry in self._entries.values():
            if entry.final:
                self._write(entry)

    def _is_complete(self, entry: Entry) -> bool:
        return entry.final and (entry.translated_final or not self._translate)

    def _write(self, entry: Entry) -> None:
        if self.path is None or entry.utterance_id in self._written:
            return
        self._written.add(entry.utterance_id)
        lines = [f"[{entry.clock}] {entry.original}"]
        if entry.translation:
            lines.append(f"           → {entry.translation}")
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            log.error("Cannot write transcript %s: %s", self.path, exc)
