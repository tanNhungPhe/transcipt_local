# Live Translate — phụ đề & dịch tự động thời gian thực

Nghe **âm thanh đang phát trên máy** (YouTube, phim, Zoom, Google Meet, Teams, Messenger…) hoặc **micro**,
hiện transcript ngay khi người nói đang nói và dịch sang tiếng Việt (hoặc ngôn ngữ khác) trong một cửa sổ
phụ đề luôn nằm trên cùng.

## Chạy

Nhấp đúp `run.bat` (lần đầu sẽ tự cài thư viện), hoặc:

```
.venv\Scripts\python.exe main.py                     # mở cửa sổ phụ đề
.venv\Scripts\python.exe main.py --source mic         # nghe micro thay vì loa
.venv\Scripts\python.exe main.py --lang en --target vi
.venv\Scripts\python.exe main.py file video.mp4       # xuất video.en.srt + video.vi.srt
```

Bấm **▶ Bắt đầu** rồi phát video hoặc vào cuộc gọi.

- Chữ **vàng nghiêng** = câu đang nói (cập nhật ~0.5s/lần); chữ **trắng** = câu đã chốt.
- **Chế độ phụ đề**: chỉ hiện 2 câu gần nhất, chữ to — đặt cửa sổ dưới video. `A−/A+` đổi cỡ chữ, thanh **Mờ** đổi độ trong suốt.
- Transcript mỗi phiên tự lưu vào `transcripts/transcript_YYYYmmdd_HHMMSS.txt` (nút *Thư mục transcript*).
- **Phiên âm file video…**: chọn file mp4/mkv/mp3… → tạo phụ đề `.srt` gốc và bản dịch cạnh file.

## Mẹo

- Chọn đúng ngôn ngữ ở ô **Nói** (thay vì *Tự nhận diện*) sẽ nhanh và chính xác hơn.
- Máy yếu / không có GPU NVIDIA: chọn model `small` hoặc `base`.
- Cuộc gọi: chọn *Âm thanh hệ thống* để dịch giọng người bên kia; chọn *Micro* để phiên âm giọng của bạn.
- Dịch dùng Google Translate (cần Internet). Nhận dạng giọng nói chạy hoàn toàn trên máy.

## Kiến trúc

```
WASAPI loopback/mic ─► 16 kHz mono ─► Silero VAD ─► StreamingSegmenter
                                                    │ partial (0.5s) ─► Whisper beam 1 ─► UI (vàng)
                                                    └ final (im lặng 0.6s) ─► Whisper beam 5 ─► UI (trắng)
                                                                                   └► TranslationWorker ─► UI
```

| File | Vai trò |
|---|---|
| `livetrans/audio_capture.py` | Thu âm loa (loopback) / micro qua WASAPI |
| `livetrans/segmenter.py` | Cắt câu theo VAD, phát partial định kỳ |
| `livetrans/transcriber.py` | faster-whisper (CUDA tự động), lọc ảo giác |
| `livetrans/translator.py` | Dịch + cache + worker gộp các partial |
| `livetrans/engine.py` | Ghép pipeline real-time |
| `livetrans/overlay.py` | Cửa sổ phụ đề tkinter |
| `livetrans/file_mode.py` | Video/audio → `.srt` |

## Test

```
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe tests\bench_realtime.py speech.wav   # đo độ trễ
```
