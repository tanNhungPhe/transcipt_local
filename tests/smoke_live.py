"""End-to-end smoke test: play a WAV through the speakers and capture it via loopback.

    python tests/smoke_live.py speech.wav
"""
import sys
import threading
import time
import winsound
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livetrans.config import Settings  # noqa: E402
from livetrans.engine import LiveEngine, UiEvent  # noqa: E402
from livetrans.transcriber import get_transcriber  # noqa: E402


def main() -> None:
    wav = sys.argv[1]
    get_transcriber("large-v3-turbo")  # preload so timing reflects steady state
    started = time.perf_counter()

    def emit(ev: UiEvent) -> None:
        t = time.perf_counter() - started
        if ev.kind in ("final", "translation", "status", "error") and (ev.final or ev.kind in ("status", "error")):
            print(f"{t:5.1f}s {ev.kind:11} #{ev.utterance_id:<2} {ev.text}", flush=True)

    engine = LiveEngine(Settings(), emit)
    engine.start(Settings())
    time.sleep(2)
    player = threading.Thread(target=winsound.PlaySound, args=(wav, winsound.SND_FILENAME))
    started = time.perf_counter()
    player.start()
    player.join()
    time.sleep(3)
    engine.stop()
    time.sleep(1)


if __name__ == "__main__":
    main()
