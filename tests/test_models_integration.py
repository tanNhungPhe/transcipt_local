"""Real-model checks (skipped until setup_offline.bat has downloaded the models).

    .venv\\Scripts\\python.exe -m pytest tests -m integration
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from livetrans.model_store import has_nllb, has_ov_whisper, nllb_dir, ov_whisper_dir

ROOT = Path(__file__).resolve().parents[1]
SENTENCE = "Hello everyone, thank you for joining the meeting today."
pytestmark = pytest.mark.integration

needs_whisper = pytest.mark.skipif(not has_ov_whisper(ov_whisper_dir("small")), reason="run setup_offline.bat")
needs_nllb = pytest.mark.skipif(not has_nllb(nllb_dir()), reason="run setup_offline.bat")


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory) -> Path:
    if sys.platform != "win32":
        pytest.skip("uses Windows text-to-speech")
    path = tmp_path_factory.mktemp("tts") / "speech.wav"
    script = ("Add-Type -AssemblyName System.Speech; $t = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$t.SetOutputToWaveFile('{path}'); $t.Speak('{SENTENCE}'); $t.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60)  # noqa: S603,S607
    return path


def _load_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = int(len(pcm) * 16000 / rate)
    return np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)


@needs_nllb
def test_nllb_translates_offline():
    from livetrans.nllb import get_nllb_backend

    backend = get_nllb_backend()
    assert "chào" in backend("Hello everyone.", "vi", "en").lower()
    assert backend.translate_batch(["Good morning.", "Thank you."], "ja", "en")[1]


@needs_whisper
def test_openvino_whisper_cpu_live_and_file(speech_wav):
    from livetrans.ov_backend import OpenVinoTranscriber

    tr = OpenVinoTranscriber(ov_whisper_dir("small"), "CPU", "CPU (OpenVINO)")
    audio = _load_16k(speech_wav)
    live = tr.transcribe(audio, final=True)
    assert "meeting" in live.text.lower() and live.language == "en"
    segments, lang = tr.transcribe_long(np.concatenate([np.zeros(32000, np.float32), audio]), None, lambda m: None)
    assert lang == "en" and segments and segments[0].start >= 1.0
    assert "meeting" in " ".join(s.text for s in segments).lower()


@needs_whisper
def test_process_exits_after_using_intel_gpu():
    from livetrans.devices import intel_gpu_id, openvino_devices

    if intel_gpu_id(openvino_devices() or {}) is None:
        pytest.skip("no Intel GPU")
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
            "import numpy as np\n"
            "from livetrans.transcriber import get_transcriber\n"
            "get_transcriber('small', 'igpu').transcribe(np.zeros(16000, np.float32), final=True)\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)  # noqa: S603
    assert proc.returncode == 0, proc.stderr[-2000:]


@needs_whisper
@needs_nllb
def test_file_mode_runs_with_network_blocked(speech_wav, tmp_path):
    media = tmp_path / "talk.wav"
    media.write_bytes(speech_wav.read_bytes())
    guard = (
        "import socket, sys\n"
        "def _blocked(*a, **k):\n"
        "    sys.stderr.write(f'NETWORK ATTEMPT {a[1:]!r}\\n')\n"
        "    raise OSError('network blocked by test')\n"
        "socket.socket.connect = socket.socket.connect_ex = _blocked\n"
        "socket.create_connection = socket.getaddrinfo = _blocked\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import main\n"
        f"sys.exit(main.main(['--model', 'small', '--device', 'cpu', 'file', {str(media)!r}]))\n"
    )
    proc = subprocess.run([sys.executable, "-X", "utf8", "-c", guard], capture_output=True, text=True,  # noqa: S603
                          encoding="utf-8", timeout=600, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "NETWORK ATTEMPT" not in proc.stderr  # nothing even tried to go online
    vi = (tmp_path / "talk.vi.srt").read_text(encoding="utf-8").lower()
    assert "cuộc họp" in vi
