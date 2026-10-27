"""Streaming utterance segmenter: emits live partials and finalized utterances."""
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

SpeechScorer = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class SegmenterConfig:
    sample_rate: int = 16000
    start_prob: float = 0.5        # prob needed to start an utterance
    keep_prob: float = 0.35        # prob that counts as "still speaking"
    silence_sec: float = 0.6       # trailing silence that ends an utterance
    min_speech_sec: float = 0.25   # shorter blips are discarded
    max_utterance_sec: float = 10.0
    partial_interval_sec: float = 0.5
    preroll_sec: float = 0.3


@dataclass(frozen=True)
class SegmentEvent:
    kind: str            # "partial" | "final"
    utterance_id: int
    audio: np.ndarray
    start_sec: float
    end_sec: float


class StreamingSegmenter:
    def __init__(self, scorer: SpeechScorer, config: SegmenterConfig = SegmenterConfig()):
        self._scorer = scorer
        self._cfg = config
        self._preroll: deque[np.ndarray] = deque()
        self._preroll_len = 0
        self._chunks: list[np.ndarray] = []
        self._speaking = False
        self._speech_sec = 0.0
        self._silence_sec = 0.0
        self._since_partial = 0.0
        self._consumed = 0          # total samples seen
        self._start_sample = 0
        self._next_id = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, chunk: np.ndarray) -> list[SegmentEvent]:
        dur = len(chunk) / self._cfg.sample_rate
        prob = self._scorer(chunk)
        self._consumed += len(chunk)
        if not self._speaking:
            return self._feed_idle(chunk, prob)
        return self._feed_speaking(chunk, prob, dur)

    def flush(self) -> list[SegmentEvent]:
        if self._speaking and self._speech_sec >= self._cfg.min_speech_sec:
            return [self._finish()]
        self._reset()
        return []

    def _feed_idle(self, chunk: np.ndarray, prob: float) -> list[SegmentEvent]:
        if prob < self._cfg.start_prob:
            self._push_preroll(chunk)
            return []
        self._speaking = True
        self._chunks = list(self._preroll) + [chunk]
        self._start_sample = self._consumed - sum(len(c) for c in self._chunks)
        self._speech_sec = len(chunk) / self._cfg.sample_rate
        self._silence_sec = 0.0
        self._since_partial = 0.0
        self._clear_preroll()
        return []

    def _feed_speaking(self, chunk: np.ndarray, prob: float, dur: float) -> list[SegmentEvent]:
        self._chunks.append(chunk)
        self._since_partial += dur
        if prob >= self._cfg.keep_prob:
            self._speech_sec += dur
            self._silence_sec = 0.0
        else:
            self._silence_sec += dur

        if self._silence_sec >= self._cfg.silence_sec:
            if self._speech_sec < self._cfg.min_speech_sec:
                self._reset()
                return []
            return [self._finish()]
        if self._buffer_sec() >= self._cfg.max_utterance_sec:
            return [self._finish()]
        if self._since_partial >= self._cfg.partial_interval_sec:
            self._since_partial = 0.0
            return [self._event("partial")]
        return []

    def _finish(self) -> SegmentEvent:
        event = self._event("final")
        self._next_id += 1
        self._reset()
        return event

    def _event(self, kind: str) -> SegmentEvent:
        audio = np.concatenate(self._chunks)
        start = self._start_sample / self._cfg.sample_rate
        end = start + len(audio) / self._cfg.sample_rate
        return SegmentEvent(kind, self._next_id, audio, start, end)

    def _buffer_sec(self) -> float:
        return sum(len(c) for c in self._chunks) / self._cfg.sample_rate

    def _reset(self) -> None:
        self._speaking = False
        self._chunks = []
        self._speech_sec = 0.0
        self._silence_sec = 0.0
        self._since_partial = 0.0

    def _push_preroll(self, chunk: np.ndarray) -> None:
        self._preroll.append(chunk)
        self._preroll_len += len(chunk)
        limit = int(self._cfg.preroll_sec * self._cfg.sample_rate)
        while self._preroll and self._preroll_len - len(self._preroll[0]) >= limit:
            self._preroll_len -= len(self._preroll.popleft())

    def _clear_preroll(self) -> None:
        self._preroll.clear()
        self._preroll_len = 0
