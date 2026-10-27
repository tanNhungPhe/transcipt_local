"""Simulate a live stream from a WAV file and measure end-to-end latency.

    python tests/bench_realtime.py speech.wav [model]
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livetrans.audio_capture import CHUNK_SAMPLES  # noqa: E402
from livetrans.audio_utils import resample, to_mono  # noqa: E402
from livetrans.engine import select_events  # noqa: E402
from livetrans.segmenter import StreamingSegmenter  # noqa: E402
from livetrans.transcriber import get_transcriber  # noqa: E402
from livetrans.translator import Translator  # noqa: E402
from livetrans.vad import SileroScorer  # noqa: E402


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as wf:
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return resample(to_mono(raw, wf.getnchannels()), wf.getframerate())


def main() -> None:
    audio = load_wav(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "large-v3-turbo"
    t0 = time.perf_counter()
    tr = get_transcriber(model)
    print(f"model {model} on {tr.device} loaded in {time.perf_counter() - t0:.1f}s")
    translator = Translator("vi")
    seg = StreamingSegmenter(SileroScorer())
    asr_times = []
    lang = None
    for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        for ev in select_events(seg.feed(audio[i:i + CHUNK_SAMPLES])):
            final = ev.kind == "final"
            t = time.perf_counter()
            res = tr.transcribe(ev.audio, final=final, language_hint=None if final else lang)
            asr = (time.perf_counter() - t) * 1000
            asr_times.append(asr)
            if res.text and res.language_probability >= 0.7:
                lang = res.language
            line = f"[{ev.kind:7}] {ev.end_sec:5.1f}s asr={asr:4.0f}ms  {res.text}"
            if ev.kind == "final" and res.text:
                t = time.perf_counter()
                vi = translator.translate(res.text)
                line += f"\n{'':10}→ ({(time.perf_counter() - t) * 1000:.0f}ms) {vi}"
            print(line)
    print(f"ASR latency: median {np.median(asr_times):.0f} ms, max {max(asr_times):.0f} ms, calls {len(asr_times)}")


if __name__ == "__main__":
    main()
