"""WASAPI capture (system loopback or microphone) -> 16 kHz mono fixed-size chunks."""
import logging
import queue

import numpy as np
import pyaudiowpatch as pyaudio

from .audio_utils import TARGET_RATE, ChunkRebuffer, resample, to_mono

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 512 * 3  # 96 ms at 16 kHz, a multiple of the Silero window
SOURCE_SYSTEM = "system"
SOURCE_MIC = "mic"


class AudioCapture:
    def __init__(self, source: str = SOURCE_SYSTEM, max_queue: int = 400):
        if source not in (SOURCE_SYSTEM, SOURCE_MIC):
            raise ValueError(f"unknown audio source: {source}")
        self.source = source
        self.chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._rebuffer = ChunkRebuffer(CHUNK_SAMPLES)
        self._channels = 1
        self._rate = TARGET_RATE
        self.device_name = ""

    def _device(self, pa: pyaudio.PyAudio) -> dict:
        if self.source == SOURCE_SYSTEM:
            return pa.get_default_wasapi_loopback()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        return pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        try:
            dev = self._device(self._pa)
        except (OSError, LookupError) as exc:
            self._pa.terminate()
            self._pa = None
            raise RuntimeError(f"Không tìm thấy thiết bị âm thanh ({self.source}): {exc}") from exc
        self._channels = max(1, int(dev["maxInputChannels"]))
        self._rate = int(dev["defaultSampleRate"])
        self.device_name = dev["name"]
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=self._rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=int(self._rate * 0.05),
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        log.info("Capturing from %s (%d Hz, %d ch)", self.device_name, self._rate, self._channels)

    def _callback(self, in_data, frame_count, time_info, status):
        frames = np.frombuffer(in_data, dtype=np.float32)
        mono = resample(to_mono(frames, self._channels), self._rate, TARGET_RATE)
        for chunk in self._rebuffer.push(mono):
            try:
                self.chunks.put_nowait(chunk)
            except queue.Full:
                self.chunks.get_nowait()  # drop oldest audio rather than fall behind
                self.chunks.put_nowait(chunk)
        return (None, pyaudio.paContinue)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
