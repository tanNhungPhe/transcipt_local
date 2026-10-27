"""User-facing settings and language tables."""
from dataclasses import dataclass, replace

# Display name -> Google Translate code
TARGET_LANGUAGES: dict[str, str] = {
    "Tiếng Việt": "vi",
    "English": "en",
    "中文 (简体)": "zh-CN",
    "日本語": "ja",
    "한국어": "ko",
    "Français": "fr",
    "Deutsch": "de",
    "Español": "es",
    "Русский": "ru",
    "ไทย": "th",
}

# Display name -> Whisper code (None = auto-detect)
SOURCE_LANGUAGES: dict[str, str | None] = {
    "Tự nhận diện": None,
    "English": "en",
    "Tiếng Việt": "vi",
    "中文": "zh",
    "日本語": "ja",
    "한국어": "ko",
    "Français": "fr",
    "Deutsch": "de",
    "Español": "es",
    "Русский": "ru",
}

MODELS = ("large-v3-turbo", "medium", "small", "base", "tiny")


@dataclass(frozen=True)
class Settings:
    source: str = "system"              # "system" | "mic"
    model: str = "large-v3-turbo"
    device: str = "auto"                # "auto" | "cuda" | "cpu"
    source_language: str | None = None
    target_language: str = "vi"
    translate: bool = True
    translate_partials: bool = True
    transcript_dir: str = "transcripts"

    def with_changes(self, **changes) -> "Settings":
        return replace(self, **changes)
