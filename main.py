"""Live Translate — real-time transcripts + translation for videos and calls.

    python main.py                      # open the live subtitle window
    python main.py file video.mp4       # write video.<lang>.srt and video.vi.srt
"""
import argparse
import logging
import sys
from pathlib import Path

from livetrans.config import MODELS, Settings


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phụ đề & dịch tự động thời gian thực")
    p.add_argument("--model", default="large-v3-turbo", choices=MODELS)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--source", default="system", choices=("system", "mic"))
    p.add_argument("--lang", default=None, help="ngôn ngữ nói (en, ja, zh…); bỏ trống = tự nhận diện")
    p.add_argument("--target", default="vi", help="ngôn ngữ dịch sang (vi, en, ja…)")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")
    f = sub.add_parser("file", help="phiên âm một file video/audio ra .srt")
    f.add_argument("path", type=Path)
    return p.parse_args(argv)


def _run_file(args: argparse.Namespace) -> int:
    from livetrans.file_mode import transcribe_file
    from livetrans.transcriber import get_transcriber
    from livetrans.translator import Translator

    translator = None if args.no_translate else Translator(args.target)
    result = transcribe_file(args.path, get_transcriber(args.model, args.device), translator, args.lang,
                             progress=lambda msg: print(f"\r{msg[:100]:<100}", end="", flush=True))
    print(f"\nĐã tạo: {result.original_srt}")
    if result.translated_srt:
        print(f"Đã tạo: {result.translated_srt}")
    return 0


def main(argv: list[str]) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "file":
        return _run_file(args)
    from livetrans.overlay import run_gui

    run_gui(Settings(source=args.source, model=args.model, device=args.device, source_language=args.lang,
                     target_language=args.target, translate=not args.no_translate))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
