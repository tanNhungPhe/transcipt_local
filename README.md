# Live Translate — phụ đề & dịch tự động thời gian thực, **100% offline**

Nghe **âm thanh đang phát trên máy** (YouTube, phim, Zoom, Google Meet, Teams, Messenger…) hoặc **micro**,
hiện transcript ngay khi người nói đang nói và dịch sang tiếng Việt (hoặc ngôn ngữ khác) trong một cửa sổ
phụ đề luôn nằm trên cùng.

Nhận dạng giọng nói (Whisper) **và** dịch (NLLB-200) đều chạy trên máy. Sau khi tải model một lần,
app không gửi bất kỳ dữ liệu nào ra Internet:

- Không còn Google Translate.
- HuggingFace bị khoá ở chế độ offline (`HF_HUB_OFFLINE=1`).
- Telemetry của OpenVINO (mặc định gửi thống kê sử dụng lên Google Analytics khi `import openvino`) được tắt
  bằng cơ chế opt-out chính thức của Intel: file `%LOCALAPPDATA%\Intel Corporation\openvino_telemetry` = `0`.
- Test `tests/test_models_integration.py` chạy app với mọi socket bị chặn và kiểm tra **không có lần thử kết nối nào**.

## Cài đặt lần đầu (cần Internet một lần)

Nhấp đúp `setup_offline.bat` — tự tạo môi trường Python, cài thư viện và tải model vào thư mục `models\`
(~3 GB). Xong có thể rút mạng.

| Model | Dung lượng | Dùng cho |
|---|---|---|
| `models\whisper-ov\large-v3-turbo` | ~0.9 GB | Nhận dạng giọng nói (OpenVINO: NPU / iGPU / CPU Intel) |
| `models\whisper-ov\small` | ~0.3 GB | Bản nhẹ, nhanh hơn |
| `models\nllb-1.3B` | ~1.4 GB | Dịch offline 200 ngôn ngữ (CPU) |
| `models\whisper-ct2\…` | | Chỉ tải khi máy có GPU NVIDIA |
| `models\ov_cache\` | | Bản biên dịch sẵn cho NPU/GPU — giúp mở app nhanh |

Tuỳ chọn: `setup_offline.bat --models large-v3-turbo small base` để tải thêm cỡ model.

## Chạy

Nhấp đúp `run.bat`, hoặc:

```
.venv\Scripts\python.exe main.py                     # mở cửa sổ phụ đề
.venv\Scripts\python.exe main.py --device npu        # ép chạy trên Intel NPU
.venv\Scripts\python.exe main.py --source mic         # nghe micro thay vì loa
.venv\Scripts\python.exe main.py --lang en --target vi
.venv\Scripts\python.exe main.py file video.mp4       # xuất video.en.srt + video.vi.srt
```

Bấm **▶ Bắt đầu** rồi phát video hoặc vào cuộc gọi. Thanh trạng thái cho biết Whisper đang chạy trên thiết bị nào.

- Chữ **vàng nghiêng** = câu đang nói; chữ **trắng** = câu đã chốt.
- **Chế độ phụ đề**: chỉ hiện 2 câu gần nhất, chữ to. `A−/A+` đổi cỡ chữ, thanh **Mờ** đổi độ trong suốt.
- Transcript mỗi phiên tự lưu vào `transcripts/transcript_YYYYmmdd_HHMMSS.txt`.
- **Phiên âm file video…**: chọn file mp4/mkv/mp3… → tạo phụ đề `.srt` gốc và bản dịch cạnh file.

## Máy Intel Core Ultra (NPU) 16 GB

Ô thiết bị **Tự động** chọn theo thứ tự: NVIDIA CUDA → **Intel NPU** → Intel iGPU (Arc) → CPU.
Nếu thiết bị nào lỗi, app tự chuyển sang thiết bị kế tiếp.

- **NPU**: tiết kiệm pin, CPU rảnh để chạy bộ dịch. Lần chạy đầu tiên NPU phải biên dịch model (có thể vài phút);
  `setup_offline.bat` đã làm sẵn bước này và lưu vào `models\ov_cache`.
- **iGPU Arc**: thường nhanh hơn NPU, tốn điện hơn. Thử cả hai bằng ô chọn thiết bị.
- NPU/iGPU chỉ hỗ trợ giải mã greedy (không beam search) — app tự xử lý, độ chính xác chênh lệch rất nhỏ.
- RAM dùng khoảng 3–4 GB (Whisper turbo + NLLB). Nếu thấy trễ, chọn model `small`.
- Tham khảo đo trên i7-11800H (không NPU, iGPU UHD đời cũ — yếu hơn nhiều so với Arc/NPU của Core Ultra):
  `large-v3-turbo` ~3 s/câu trên CPU, ~4 s trên iGPU UHD; `small` ~1 s trên iGPU UHD; dịch NLLB ~0.5 s/câu.
  Đo trên máy của bạn: `tests\bench_realtime.py speech.wav large-v3-turbo npu`.
- Cập nhật driver NPU/GPU Intel mới nhất nếu NPU không xuất hiện.

## Mẹo

- Chọn đúng ngôn ngữ ở ô **Nói** (thay vì *Tự nhận diện*) sẽ nhanh và chính xác hơn.
- Cuộc gọi: chọn *Âm thanh hệ thống* để dịch giọng người bên kia; chọn *Micro* để phiên âm giọng của bạn.
- Model dịch NLLB-200 có giấy phép CC-BY-NC 4.0 (dùng cá nhân / phi thương mại).

## Kiến trúc

```
WASAPI loopback/mic ─► 16 kHz mono ─► Silero VAD ─► StreamingSegmenter
                                                    │ partial (0.5s) ─► Whisper greedy ─► UI (vàng)
                                                    └ final (im lặng 0.6s) ─► Whisper ─► UI (trắng)
                                                                                └► NLLB-200 (CPU) ─► UI
Whisper: OpenVINO (NPU / iGPU / CPU)  hoặc  faster-whisper (NVIDIA CUDA / CPU)
```

| File | Vai trò |
|---|---|
| `livetrans/audio_capture.py` | Thu âm loa (loopback) / micro qua WASAPI |
| `livetrans/segmenter.py` | Cắt câu theo VAD, phát partial định kỳ |
| `livetrans/devices.py` | Chọn thiết bị: CUDA → NPU → iGPU → CPU, kèm fallback |
| `livetrans/transcriber.py` | Giao diện chung + nạp model theo thiết bị |
| `livetrans/ov_backend.py` | Whisper trên OpenVINO (Intel NPU / iGPU / CPU) |
| `livetrans/ct2_backend.py` | Whisper trên faster-whisper (NVIDIA / CPU) |
| `livetrans/nllb.py` | Dịch offline NLLB-200 (CTranslate2) |
| `livetrans/translator.py` | Cache dịch + worker gộp các partial |
| `livetrans/model_store.py` / `downloader.py` | Vị trí model, chế độ offline, tải model một lần |
| `livetrans/engine.py` | Ghép pipeline real-time |
| `livetrans/overlay.py` | Cửa sổ phụ đề tkinter |
| `livetrans/file_mode.py` | Video/audio → `.srt` |

## Test

```
.venv\Scripts\python.exe -m pytest tests -q                     # unit + integration (nếu đã có model)
.venv\Scripts\python.exe -m pytest tests -q -m integration      # chỉ test với model thật, có test chặn mạng
.venv\Scripts\python.exe tests\bench_realtime.py speech.wav large-v3-turbo npu   # đo độ trễ
```
