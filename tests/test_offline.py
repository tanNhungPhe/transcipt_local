"""Device selection, offline translation and OpenVINO backend logic (no models needed)."""
import wave
from dataclasses import dataclass, field

import numpy as np
import pytest

from livetrans.devices import CTRANSLATE2, OPENVINO, DevicePlan, intel_gpu_id, plan_candidates
from livetrans.downloader import wanted_backends
from livetrans.file_mode import _decode_with_pyav
from livetrans.model_store import ModelMissingError
from livetrans.nllb import FALLBACK_SOURCE, nllb_code, normalize_lang, source_tokens
from livetrans.ov_backend import OpenVinoTranscriber, _to_segments, merge_regions, strip_lang_token
from livetrans.transcriber import load_first_working
from livetrans.translator import TranslationJob, TranslationWorker, Translator

INTEL_LAPTOP = {"CPU": "Intel(R) Core(TM) Ultra 5 125H", "GPU": "Intel(R) Arc(TM) Graphics (iGPU)",
                "NPU": "Intel(R) AI Boost"}
MIXED = {"CPU": "Intel i7", "GPU.0": "Intel(R) UHD Graphics (iGPU)", "GPU.1": "NVIDIA GeForce RTX 3060 (dGPU)"}


def _ids(plans):
    return [(p.backend, p.device) for p in plans]


# ---------- device selection ----------
def test_auto_on_core_ultra_prefers_npu_then_igpu_then_cpu():
    assert _ids(plan_candidates("auto", False, INTEL_LAPTOP)) == [
        (OPENVINO, "NPU"), (OPENVINO, "GPU"), (OPENVINO, "CPU"), (CTRANSLATE2, "cpu")]


def test_auto_prefers_cuda_when_present():
    assert _ids(plan_candidates("auto", True, MIXED))[0] == (CTRANSLATE2, "cuda")


def test_igpu_skips_nvidia_listed_by_openvino():
    assert intel_gpu_id(MIXED) == "GPU.0"
    assert intel_gpu_id({"GPU.0": "NVIDIA GeForce"}) is None


def test_npu_request_without_npu_falls_back():
    assert _ids(plan_candidates("npu", False, MIXED))[0] == (OPENVINO, "GPU.0")


def test_without_openvino_only_ctranslate2_cpu():
    assert _ids(plan_candidates("auto", False, None)) == [(CTRANSLATE2, "cpu")]


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        plan_candidates("tpu", False, None)


def test_load_first_working_falls_through_failures():
    plans = [DevicePlan(OPENVINO, "NPU", "NPU"), DevicePlan(OPENVINO, "CPU", "CPU")]

    def loader(plan, size):
        if plan.device == "NPU":
            raise RuntimeError("compile failed")
        return f"{size}@{plan.device}"

    assert load_first_working("small", plans, loader) == "small@CPU"


def test_load_first_working_reports_every_device():
    def loader(plan, size):
        raise ModelMissingError("run setup") if plan.device == "CPU" else RuntimeError("compile failed")

    plans = [DevicePlan(OPENVINO, "NPU", "NPU"), DevicePlan(OPENVINO, "CPU", "CPU")]
    with pytest.raises(ModelMissingError, match="compile failed.*run setup"):
        load_first_working("small", plans, loader)


@pytest.mark.parametrize("backend,has_ov,has_cuda,expected", [
    ("auto", True, False, (True, False)),
    ("auto", True, True, (True, True)),
    ("auto", False, False, (False, True)),
    ("openvino", False, True, (True, False)),
    ("both", False, False, (True, True)),
])
def test_wanted_backends(backend, has_ov, has_cuda, expected):
    assert wanted_backends(backend, has_ov, has_cuda) == expected


# ---------- NLLB helpers ----------
def test_language_codes():
    assert normalize_lang("<|zh|>") == "zh"
    assert normalize_lang("zh-CN") == "zh"
    assert nllb_code("vi") == "vie_Latn"
    assert nllb_code("xx") is None
    assert source_tokens(["▁Hi"], "eng_Latn") == ["eng_Latn", "▁Hi", "</s>"]
    assert FALLBACK_SOURCE == "eng_Latn"


def test_translator_skips_same_language_and_keys_cache_by_source():
    calls = []
    t = Translator("vi", backend=lambda text, tgt, src: calls.append((text, src)) or f"[{src}]{text}")
    assert t.translate("xin chào", "vi") == "xin chào"
    assert t.translate("hi", "en") == "[en]hi"
    assert t.translate("hi", "fr") == "[fr]hi"
    assert t.translate("hi", "en") == "[en]hi"
    assert calls == [("hi", "en"), ("hi", "fr")]


@dataclass
class _BatchBackend:
    calls: list = field(default_factory=list)

    def __call__(self, text, target, source=None):
        return self.translate_batch([text], target, source)[0]

    def translate_batch(self, texts, target, source=None):
        self.calls.append(list(texts))
        return [t.upper() for t in texts]


def test_translate_many_batches_and_keeps_blank_lines():
    backend = _BatchBackend()
    t = Translator("vi", backend=backend)
    assert t.translate_many(["a", " ", "b"], "en") == ["A", "", "B"]
    assert backend.calls == [["a", "b"]]
    assert t.translate_many(["giữ nguyên"], "vi") == ["giữ nguyên"]


def test_worker_passes_source_language():
    import threading
    import time

    seen, done = [], threading.Event()
    w = TranslationWorker(Translator("vi", lambda text, tgt, src: seen.append(src) or text),
                          lambda job, tr: done.set(), lambda m: None)
    w.start()
    w.submit(TranslationJob(0, "hello", True, "en"))
    done.wait(2)
    w.stop()
    time.sleep(0.05)
    assert seen == ["en"]


# ---------- OpenVINO backend (fake pipeline) ----------
@dataclass
class _Chunk:
    start_ts: float
    end_ts: float
    text: str


@dataclass
class _Result:
    texts: list
    language: str = "en"
    chunks: list = field(default_factory=list)


class _FakePipe:
    """Rejects beam search like the Intel iGPU / NPU plugins do."""

    def __init__(self):
        self.calls = []

    def generate(self, audio, **kwargs):
        self.calls.append(kwargs)
        if "num_beams" in kwargs:
            raise RuntimeError("Not Implemented")
        return _Result([" Hello world."])


def test_ov_transcriber_disables_only_the_rejected_feature():
    pipe = _FakePipe()
    tr = OpenVinoTranscriber(None, "GPU.0", "Intel iGPU", pipeline=pipe)
    res = tr.transcribe(np.zeros(8000, np.float32), final=True, prompt="ctx")
    assert (res.text, res.language) == ("Hello world.", "en")
    assert pipe.calls[-1] == {"initial_prompt": "ctx"}  # prompt kept, beams dropped
    tr.transcribe(np.zeros(8000, np.float32), final=True, prompt="more")
    assert pipe.calls[-1] == {"initial_prompt": "more"}  # no second beam attempt
    assert len([c for c in pipe.calls if "num_beams" in c]) == 2


def test_ov_transcriber_reraises_when_nothing_helps():
    class Broken(_FakePipe):
        def generate(self, audio, **kwargs):
            if self.calls:
                raise RuntimeError("device lost")
            self.calls.append(kwargs)
            return _Result([""])

    tr = OpenVinoTranscriber(None, "NPU", "Intel NPU", pipeline=Broken())
    with pytest.raises(RuntimeError, match="device lost"):
        tr.transcribe(np.zeros(8000, np.float32), final=True, prompt="x")


def test_transcribe_long_votes_language_across_windows(monkeypatch):
    import livetrans.ov_backend as ovb

    class Multi(_FakePipe):
        def generate(self, audio, **kwargs):
            self.calls.append(kwargs)
            lang = "en" if len(audio) == 16000 * 3 else "ja"
            return _Result([" Hi."], language=lang, chunks=[_Chunk(0.0, 0.5, " Hi.")])

    monkeypatch.setattr(ovb, "speech_windows", lambda audio: [(0, 16000), (16000, 64000), (64000, 80000)])
    tr = OpenVinoTranscriber(None, "CPU", "CPU", pipeline=Multi())
    segments, lang = tr.transcribe_long(np.zeros(80000, np.float32), None, lambda m: None)
    assert lang == "en"  # the 3 s window outweighs two 1 s windows
    assert [s.start for s in segments] == [0.0, 1.0, 4.0]


def test_ov_transcriber_passes_language_token():
    pipe = _FakePipe()
    tr = OpenVinoTranscriber(None, "CPU", "CPU", language="ja", pipeline=pipe)
    tr.transcribe(np.zeros(8000, np.float32), final=False)
    assert pipe.calls[-1]["language"] == "<|ja|>"


def test_merge_regions_respects_gap_and_length():
    assert merge_regions([(0, 10), (12, 20), (40, 50)], max_len=100, max_gap=5) == [(0, 20), (40, 50)]
    assert merge_regions([(0, 60), (62, 120)], max_len=100, max_gap=5) == [(0, 60), (62, 120)]
    assert merge_regions([], 100, 5) == []


def test_chunks_are_offset_and_unterminated_chunk_closed():
    result = _Result(["x"], chunks=[_Chunk(0.0, 1.5, " Hi."), _Chunk(2.0, -1.0, " Bye now.")])
    segs = _to_segments(result, offset=10.0, window_end=14.0)
    assert [(s.start, s.end, s.text) for s in segs] == [(10.0, 11.5, "Hi."), (12.0, 14.0, "Bye now.")]


def test_no_chunks_gives_one_cue_per_window():
    segs = _to_segments(_Result([" Whole window."]), offset=3.0, window_end=9.0)
    assert [(s.start, s.end, s.text) for s in segs] == [(3.0, 9.0, "Whole window.")]


def test_strip_lang_token():
    assert strip_lang_token("<|vi|>") == "vi"
    assert strip_lang_token("zh-CN") == "zh"
    assert strip_lang_token(None) == ""


# ---------- OpenVINO telemetry opt-out ----------
@pytest.mark.parametrize("existing", [None, "", "1", "0"])
def test_openvino_telemetry_opted_out(tmp_path, existing):
    from livetrans.model_store import opt_out_openvino_telemetry

    path = tmp_path / "Intel Corporation" / "openvino_telemetry"
    if existing is not None:
        path.parent.mkdir(parents=True)
        path.write_text(existing)
    opt_out_openvino_telemetry(path)
    assert path.read_text() == "0"


# ---------- media decoding without ffmpeg ----------
def test_pyav_fallback_decodes_wav_to_16k_mono(tmp_path):
    path = tmp_path / "tone.wav"
    rate = 44100
    t = np.arange(rate) / rate
    pcm = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.repeat(pcm, 2).tobytes())
    audio = _decode_with_pyav(path)
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16000) < 400
    assert 0.3 < float(np.abs(audio).max()) < 0.8  # FFmpeg downmixes identical L/R at -3 dB each
