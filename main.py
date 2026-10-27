"""Live Translate — real-time transcripts + translation for videos and calls, fully offline.

    python main.py                      # open the live subtitle window
    python main.py file video.mp4       # write video.<lang>.srt and video.vi.srt
    python main.py download             # one-time model download (needs Internet)
"""
import argparse
import logging
import sys
from pathlib import Path

from livetrans.config import MODELS, Settings
from livetrans.devices import DEVICES
from livetrans.model_store import enable_offline_mode, opt_out_openvino_telemetry


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phụ đề & dịch tự động thời gian thực (offline)")
    p.add_argument("--model", default="large-v3-turbo", choices=MODELS)
    p.add_argument("--device", default="auto", choices=DEVICES,
                   help="auto = NVIDIA → Intel NPU → Intel iGPU → CPU")
    p.add_argument("--source", default="system", choices=("system", "mic"))
    p.add_argument("--lang", default=None, help="ngôn ngữ nói (en, ja, zh…); bỏ trống = tự nhận diện")
    p.add_argument("--target", default="vi", help="ngôn ngữ dịch sang (vi, en, ja…)")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")
    f = sub.add_parser("file", help="phiên âm một file video/audio ra .srt")
    f.add_argument("path", type=Path)
    d = sub.add_parser("download", help="tải model một lần để dùng offline (cần Internet)")
    d.add_argument("--models", nargs="+", default=["large-v3-turbo", "small"], choices=MODELS)
    d.add_argument("--backend", default="auto", choices=("auto", "openvino", "ctranslate2", "both"))
    d.add_argument("--no-warmup", action="store_true", help="bỏ qua bước biên dịch sẵn cho NPU/GPU")
    return p.parse_args(argv)


def _run_download(args: argparse.Namespace) -> int:
    from livetrans.downloader import download_models, warm_up

    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    opt_out_openvino_telemetry()  # downloading is the only online step; usage stats still stay home
    download_models(args.models, args.backend)
    if not args.no_warmup:
        warm_up(args.models[0], args.device)
    print("Xong. Từ giờ app chạy hoàn toàn offline.")
    return 0


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
    if args.command == "download":
        return _run_download(args)
    enable_offline_mode()
    if args.command == "file":
        return _run_file(args)
    from livetrans.overlay import run_gui

    run_gui(Settings(source=args.source, model=args.model, device=args.device, source_language=args.lang,
                     target_language=args.target, translate=not args.no_translate))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
