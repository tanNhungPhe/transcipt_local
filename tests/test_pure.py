import threading
import time

import numpy as np
import pytest

from livetrans.audio_utils import ChunkRebuffer, resample, rms, to_mono
from livetrans.engine import select_events
from livetrans.segmenter import SegmentEvent, SegmenterConfig, StreamingSegmenter
from livetrans.srt import Cue, build_srt, format_timestamp
from livetrans.text_filters import collapse_repeats, is_hallucination
from livetrans.transcript_store import TranscriptStore
from livetrans.translator import TranslationJob, TranslationWorker, Translator, parse_gtx_response

CHUNK = 1600  # 0.1 s at 16 kHz


# ---------- audio utils ----------
def test_to_mono_averages_channels():
    stereo = np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float32)
    assert to_mono(stereo, 2).tolist() == [0.5, 0.5]


def test_resample_48k_to_16k_length():
    assert len(resample(np.zeros(48000, dtype=np.float32), 48000, 16000)) == 16000


def test_rms_of_empty_is_zero():
    assert rms(np.zeros(0)) == 0.0


def test_rebuffer_emits_fixed_chunks_and_keeps_remainder():
    rb = ChunkRebuffer(4)
    assert [len(c) for c in rb.push(np.ones(10))] == [4, 4]
    assert [len(c) for c in rb.push(np.ones(2))] == [4]


def test_rebuffer_rejects_bad_size():
    with pytest.raises(ValueError):
        ChunkRebuffer(0)


# ---------- segmenter ----------
def _speech(n):
    return [np.full(CHUNK, 0.5, dtype=np.float32)] * n


def _silence(n):
    return [np.zeros(CHUNK, dtype=np.float32)] * n


def _feed(seg, chunks):
    return [ev for c in chunks for ev in seg.feed(c)]


@pytest.fixture
def seg():
    scorer = lambda chunk: 1.0 if chunk.max() > 0.1 else 0.0  # noqa: E731
    return StreamingSegmenter(scorer, SegmenterConfig(silence_sec=0.3, partial_interval_sec=0.2,
                                                      max_utterance_sec=2.0, preroll_sec=0.2))


def test_segmenter_emits_partials_then_final(seg):
    events = _feed(seg, _silence(5) + _speech(10) + _silence(4))
    kinds = [e.kind for e in events]
    assert "partial" in kinds
    assert kinds[-1] == "final"
    final = events[-1]
    assert final.utterance_id == 0
    assert final.start_sec == pytest.approx(0.3)  # 2 preroll chunks before speech at 0.5 s


def test_segmenter_drops_short_blip(seg):
    cfg = SegmenterConfig(min_speech_sec=0.25, silence_sec=0.3)
    s = StreamingSegmenter(lambda c: 1.0 if c.max() > 0.1 else 0.0, cfg)
    assert _feed(s, _speech(1) + _silence(5)) == []


def test_segmenter_forces_final_at_max_length(seg):
    finals = [e for e in _feed(seg, _speech(45)) if e.kind == "final"]
    assert len(finals) == 2
    assert finals[0].utterance_id == 0 and finals[1].utterance_id == 1


def test_segmenter_flush_finishes_open_utterance(seg):
    _feed(seg, _speech(5))
    assert [e.kind for e in seg.flush()] == ["final"]
    assert seg.flush() == []


def test_select_events_drops_stale_partials():
    mk = lambda k, i: SegmentEvent(k, i, np.zeros(1), 0, 0)  # noqa: E731
    events = [mk("partial", 0), mk("partial", 0), mk("final", 0), mk("partial", 1)]
    assert [(e.kind, e.utterance_id) for e in select_events(events)] == [("final", 0), ("partial", 1)]
    assert select_events([mk("partial", 0), mk("final", 0)])[0].kind == "final"


# ---------- srt / filters ----------
def test_format_timestamp():
    assert format_timestamp(3725.5) == "01:02:05,500"
    assert format_timestamp(-1) == "00:00:00,000"


def test_build_srt_skips_empty_and_renumbers():
    out = build_srt([Cue(0, 1, "a"), Cue(1, 2, "  "), Cue(2, 3, "b")])
    assert out == "1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n00:00:02,000 --> 00:00:03,000\nb\n"


@pytest.mark.parametrize("text", ["", "...", "Thank you for watching!", "Hãy subscribe cho kênh"])
def test_hallucinations_detected(text):
    assert is_hallucination(text)


def test_real_sentence_not_hallucination():
    assert not is_hallucination("Thank you for joining the meeting, let's look at the budget.")


def test_collapse_repeats():
    assert collapse_repeats("ha ha ha ha ha ha") == "ha ha ha"


# ---------- translator ----------
def test_translator_caches():
    calls = []
    t = Translator("vi", backend=lambda text, tgt: calls.append(text) or f"{tgt}:{text}")
    assert t.translate("hello") == "vi:hello"
    assert t.translate(" hello ") == "vi:hello"
    assert t.translate("") == ""
    assert calls == ["hello"]


def test_parse_gtx_response():
    data = [[["Xin chào. ", "Hello. ", None], ["Tạm biệt", "Bye", None]], None, "en"]
    assert parse_gtx_response(data) == "Xin chào. Tạm biệt"
    with pytest.raises(ValueError):
        parse_gtx_response({"error": 1})


def test_worker_coalesces_partials_and_keeps_finals():
    gate = threading.Event()
    results = []

    def slow(text, tgt):
        gate.wait(2)
        return text.upper()

    w = TranslationWorker(Translator("vi", slow), lambda job, tr: results.append((job.final, tr)), lambda m: None)
    w.start()
    w.submit(TranslationJob(0, "a", False))
    time.sleep(0.05)  # worker now blocked on "a"
    for text in ("b", "c", "d"):
        w.submit(TranslationJob(0, text, False))
    w.submit(TranslationJob(0, "final", True))
    gate.set()
    deadline = time.time() + 2
    while len(results) < 2 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    w.stop()
    assert results == [(False, "A"), (True, "FINAL")]  # b/c/d superseded by the final


# ---------- transcript store ----------
def test_store_merges_events_and_writes_log(tmp_path):
    store = TranscriptStore(tmp_path, translate=True)
    store.apply("partial", 0, "hel", False)
    store.apply("translation", 0, "xin", False)
    store.apply("final", 0, "hello", True)
    store.apply("translation", 0, "xin chào", True)
    assert not store.apply("translation", 0, "late partial", False)
    assert not store.apply("partial", 0, "stale", False)
    entry = store.entries()[0]
    assert (entry.original, entry.translation, entry.final) == ("hello", "xin chào", True)
    log = store.path.read_text(encoding="utf-8")
    assert "hello" in log and "xin chào" in log


def test_store_removes_empty_final():
    store = TranscriptStore(None, translate=False)
    store.apply("partial", 3, "noise", False)
    store.apply("final", 3, "", True)
    assert store.entries() == []
