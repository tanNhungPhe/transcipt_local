"""Offline translation with Meta NLLB-200 (distilled 1.3B, int8) on CTranslate2."""
import logging
import os
import threading
from pathlib import Path

from .model_store import has_nllb, missing, nllb_dir

log = logging.getLogger(__name__)

# ISO 639-1 (Whisper / UI codes) -> NLLB-200 language token
NLLB_CODES: dict[str, str] = {
    "en": "eng_Latn", "vi": "vie_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "th": "tha_Thai",
    "id": "ind_Latn", "ms": "zsm_Latn", "pt": "por_Latn", "it": "ita_Latn", "nl": "nld_Latn",
    "tr": "tur_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl", "ar": "arb_Arab", "hi": "hin_Deva",
    "km": "khm_Khmr", "lo": "lao_Laoo", "my": "mya_Mymr", "tl": "tgl_Latn", "yue": "yue_Hant",
    "cs": "ces_Latn", "sv": "swe_Latn", "fi": "fin_Latn", "da": "dan_Latn", "no": "nob_Latn",
    "el": "ell_Grek", "he": "heb_Hebr", "hu": "hun_Latn", "ro": "ron_Latn", "fa": "pes_Arab",
    "bn": "ben_Beng", "ta": "tam_Taml", "ur": "urd_Arab", "ca": "cat_Latn", "bg": "bul_Cyrl",
    "sr": "srp_Cyrl", "hr": "hrv_Latn", "sk": "slk_Latn", "sl": "slv_Latn", "lt": "lit_Latn",
    "lv": "lvs_Latn", "et": "est_Latn", "sw": "swh_Latn", "te": "tel_Telu", "mr": "mar_Deva",
}
FALLBACK_SOURCE = "eng_Latn"  # NLLB needs a source tag; it copes well with a wrong one
_EOS = "</s>"
_BEAM_SIZE = 2
_MAX_DECODE = 256
_BATCH = 16


def normalize_lang(lang: str | None) -> str:
    """'<|zh|>' / 'zh-CN' / 'ZH' -> 'zh'."""
    return (lang or "").strip("<|>").split("-")[0].lower()


def nllb_code(lang: str | None) -> str | None:
    return NLLB_CODES.get(normalize_lang(lang))


def source_tokens(pieces: list[str], src_code: str) -> list[str]:
    return [src_code, *pieces, _EOS]


class NllbBackend:
    def __init__(self, model_dir: Path, threads: int | None = None):
        import ctranslate2
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self._model = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8",
                                             intra_threads=threads)

    def __call__(self, text: str, target: str, source: str | None = None) -> str:
        return self.translate_batch([text], target, source)[0]

    def translate_batch(self, texts: list[str], target: str, source: str | None = None) -> list[str]:
        tgt = nllb_code(target)
        if tgt is None:
            raise ValueError(f"Không hỗ trợ dịch sang '{target}'")
        src = nllb_code(source) or FALLBACK_SOURCE
        out: list[str] = []
        for i in range(0, len(texts), _BATCH):
            batch = [source_tokens(self._tok.encode(t, add_special_tokens=False).tokens, src)
                     for t in texts[i:i + _BATCH]]
            results = self._model.translate_batch(batch, target_prefix=[[tgt]] * len(batch),
                                                  beam_size=_BEAM_SIZE, max_decoding_length=_MAX_DECODE)
            out.extend(self._decode(r.hypotheses[0][1:]) for r in results)  # drop the target-language tag
        return out

    def _decode(self, pieces: list[str]) -> str:
        ids = [i for i in (self._tok.token_to_id(p) for p in pieces) if i is not None]
        return self._tok.decode(ids, skip_special_tokens=True).strip()


_instance: NllbBackend | None = None
_instance_lock = threading.Lock()


def get_nllb_backend() -> NllbBackend:
    """Load once (~1.5 GB RAM) and share between live and file modes."""
    global _instance
    with _instance_lock:
        if _instance is None:
            path = nllb_dir()
            if not has_nllb(path):
                raise missing("dịch NLLB-200", path)
            log.info("Loading NLLB translation model from %s", path)
            _instance = NllbBackend(path)
        return _instance
