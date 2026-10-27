"""Always-on-top subtitle window with controls (tkinter)."""
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import MODELS, SOURCE_LANGUAGES, TARGET_LANGUAGES, Settings
from .engine import LiveEngine, UiEvent
from .transcript_store import TranscriptStore

BG = "#111418"
FG_ORIG = "#9aa4b2"
FG_TRANS = "#ffffff"
FG_LIVE = "#f5d76e"
SOURCES = {"Âm thanh hệ thống (video / cuộc gọi)": "system", "Micro": "mic"}
COMPACT_ENTRIES = 2
FULL_ENTRIES = 300
POLL_MS = 40


def _key_for(mapping: dict, value) -> str:
    return next((k for k, v in mapping.items() if v == value), next(iter(mapping)))


class OverlayApp:
    def __init__(self, root: tk.Tk, settings: Settings):
        self.root = root
        self.settings = settings
        self.events: queue.Queue[UiEvent] = queue.Queue()
        self.engine = LiveEngine(settings, self.events.put)
        self.store = TranscriptStore(None, settings.translate)
        self.font_size = 18
        self._build_window()
        self._build_toolbar()
        self._build_body()
        self.root.after(POLL_MS, self._poll)

    # ---------- layout ----------
    def _build_window(self) -> None:
        r = self.root
        r.title("Live Translate")
        r.configure(bg=BG)
        w, h = 960, 300
        x = (r.winfo_screenwidth() - w) // 2
        y = r.winfo_screenheight() - h - 80
        r.geometry(f"{w}x{h}+{x}+{y}")
        r.attributes("-topmost", True)
        r.attributes("-alpha", 0.92)
        r.protocol("WM_DELETE_WINDOW", self._on_close)
        style = ttk.Style(r)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#1d232b", background="#2a313a", foreground="#e6e6e6",
                        arrowcolor="#e6e6e6", bordercolor="#2a313a")
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#1d232b"), ("disabled", "#161a1f")],
                  foreground=[("disabled", "#6b7480"), ("readonly", "#e6e6e6")],
                  selectbackground=[("readonly", "#1d232b")],
                  selectforeground=[("readonly", "#e6e6e6")])
        r.option_add("*TCombobox*Listbox.background", "#1d232b")
        r.option_add("*TCombobox*Listbox.foreground", "#e6e6e6")
        r.option_add("*TCombobox*Listbox.selectBackground", "#2e7d32")
        style.configure("TCheckbutton", background=BG, foreground="#cfd6df")
        style.configure("TLabel", background=BG, foreground="#cfd6df")
        style.configure("TFrame", background=BG)

    def _combo(self, parent, values, current: str, width: int) -> tk.StringVar:
        var = tk.StringVar(value=current)
        box = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly", width=width)
        box.pack(side="left", padx=3)
        self._lockable.append(box)
        return var

    def _build_toolbar(self) -> None:
        s = self.settings
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=(6, 0))
        self._lockable: list[ttk.Combobox] = []
        self.start_btn = tk.Button(bar, text="▶ Bắt đầu", command=self._toggle, bg="#2e7d32", fg="white",
                                   relief="flat", padx=10, font=("Segoe UI", 10, "bold"))
        self.start_btn.pack(side="left")
        self.source_var = self._combo(bar, SOURCES, _key_for(SOURCES, s.source), 30)
        ttk.Label(bar, text="Nói:").pack(side="left", padx=(6, 0))
        self.src_lang_var = self._combo(bar, SOURCE_LANGUAGES, _key_for(SOURCE_LANGUAGES, s.source_language), 12)
        ttk.Label(bar, text="Dịch →").pack(side="left", padx=(6, 0))
        self.tgt_var = self._combo(bar, TARGET_LANGUAGES, _key_for(TARGET_LANGUAGES, s.target_language), 12)
        self.model_var = self._combo(bar, MODELS, s.model, 14)
        self.translate_var = tk.BooleanVar(value=s.translate)
        ttk.Checkbutton(bar, text="Dịch", variable=self.translate_var).pack(side="left", padx=4)

        bar2 = ttk.Frame(self.root)
        bar2.pack(fill="x", padx=8, pady=(4, 0))
        self.show_orig_var = tk.BooleanVar(value=True)
        self.compact_var = tk.BooleanVar(value=False)
        self.topmost_var = tk.BooleanVar(value=True)
        for text, var, cmd in (("Hiện câu gốc", self.show_orig_var, self._render),
                               ("Chế độ phụ đề", self.compact_var, self._render),
                               ("Luôn trên cùng", self.topmost_var, self._apply_topmost)):
            ttk.Checkbutton(bar2, text=text, variable=var, command=cmd).pack(side="left", padx=(0, 8))
        for text, delta in (("A−", -2), ("A+", 2)):
            tk.Button(bar2, text=text, command=lambda d=delta: self._zoom(d), bg="#1d232b", fg="white",
                      relief="flat", width=3).pack(side="left", padx=1)
        ttk.Label(bar2, text="Mờ").pack(side="left", padx=(8, 2))
        alpha = tk.Scale(bar2, from_=30, to=100, orient="horizontal", showvalue=False, length=90,
                         bg=BG, highlightthickness=0, troughcolor="#1d232b",
                         command=lambda v: self.root.attributes("-alpha", int(v) / 100))
        alpha.set(92)
        alpha.pack(side="left")
        for text, cmd in (("Phiên âm file video…", self._transcribe_file),
                          ("Thư mục transcript", self._open_transcripts)):
            tk.Button(bar2, text=text, command=cmd, bg="#1d232b", fg="white", relief="flat",
                      padx=8).pack(side="right", padx=2)
        self.status_var = tk.StringVar(value="Sẵn sàng — bấm ▶ Bắt đầu rồi phát video hoặc vào cuộc gọi")
        ttk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9)).pack(fill="x", padx=10)

    def _build_body(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.text = tk.Text(frame, bg=BG, fg=FG_TRANS, wrap="word", relief="flat", highlightthickness=0,
                            padx=8, pady=4, cursor="arrow")
        scroll = ttk.Scrollbar(frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self._apply_fonts()

    def _apply_fonts(self) -> None:
        f = self.font_size
        self.text.tag_configure("orig", foreground=FG_ORIG, font=("Segoe UI", max(9, f - 6)))
        self.text.tag_configure("trans", foreground=FG_TRANS, font=("Segoe UI", f, "bold"))
        self.text.tag_configure("live_orig", foreground=FG_ORIG, font=("Segoe UI", max(9, f - 6), "italic"))
        self.text.tag_configure("live_trans", foreground=FG_LIVE, font=("Segoe UI", f, "bold italic"))
        self.text.tag_configure("gap", font=("Segoe UI", 4))

    # ---------- actions ----------
    def _current_settings(self) -> Settings:
        return self.settings.with_changes(
            source=SOURCES[self.source_var.get()],
            source_language=SOURCE_LANGUAGES[self.src_lang_var.get()],
            target_language=TARGET_LANGUAGES[self.tgt_var.get()],
            model=self.model_var.get(),
            translate=self.translate_var.get(),
        )

    def _toggle(self) -> None:
        if self.engine.running:
            self.engine.stop()
            self.start_btn.configure(text="…", state="disabled")
            return
        self.settings = self._current_settings()
        self.store = TranscriptStore(Path(self.settings.transcript_dir), self.settings.translate)
        self._render()
        self.engine.start(self.settings)
        self.start_btn.configure(text="■ Dừng", bg="#c62828")
        for box in self._lockable:
            box.configure(state="disabled")

    def _on_stopped(self) -> None:
        self.store.flush()
        self.start_btn.configure(text="▶ Bắt đầu", bg="#2e7d32", state="normal")
        for box in self._lockable:
            box.configure(state="readonly")

    def _zoom(self, delta: int) -> None:
        self.font_size = min(48, max(10, self.font_size + delta))
        self._apply_fonts()

    def _apply_topmost(self) -> None:
        self.root.attributes("-topmost", self.topmost_var.get())

    def _open_transcripts(self) -> None:
        folder = Path(self.settings.transcript_dir).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)  # noqa: S606 - opens Explorer on a local folder we created

    def _transcribe_file(self) -> None:
        path = filedialog.askopenfilename(title="Chọn video / audio", filetypes=[
            ("Media", "*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac *.ogg"), ("Tất cả", "*.*")])
        if path:
            threading.Thread(target=self._run_file_job, args=(Path(path), self._current_settings()),
                             daemon=True).start()

    def _run_file_job(self, path: Path, s: Settings) -> None:
        from .file_mode import transcribe_file
        from .transcriber import get_transcriber
        from .translator import Translator

        status = lambda msg: self.events.put(UiEvent("status", text=f"[{path.name}] {msg}"))  # noqa: E731
        try:
            status(f"Đang tải model {s.model}…")
            result = transcribe_file(path, get_transcriber(s.model, s.device),
                                     Translator(s.target_language) if s.translate else None,
                                     s.source_language, status)
            outputs = "\n".join(str(p) for p in (result.original_srt, result.translated_srt) if p)
            self.events.put(UiEvent("file_done", text=f"{result.cue_count} câu ({result.language}):\n{outputs}"))
        except Exception as exc:  # noqa: BLE001 - report any failure to the user
            self.events.put(UiEvent("error", text=f"Phiên âm file thất bại: {exc}"))

    # ---------- event loop ----------
    def _poll(self) -> None:
        changed = False
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            changed |= self._handle(ev)
        if changed:
            self._render()
        self.root.after(POLL_MS, self._poll)

    def _handle(self, ev: UiEvent) -> bool:
        if ev.kind == "status":
            self.status_var.set(ev.text)
            if ev.text == "Đã dừng":
                self._on_stopped()
            return False
        if ev.kind == "error":
            self.status_var.set(f"⚠ {ev.text}")
            return False
        if ev.kind == "file_done":
            messagebox.showinfo("Đã tạo phụ đề", ev.text)
            return False
        if ev.kind in ("partial", "final") and ev.latency_ms:
            self.status_var.set(f"Đang nghe · nhận dạng {ev.latency_ms} ms")
        return self.store.apply(ev.kind, ev.utterance_id, ev.text, ev.final)

    def _render(self) -> None:
        at_bottom = self.text.yview()[1] > 0.98
        limit = COMPACT_ENTRIES if self.compact_var.get() else FULL_ENTRIES
        show_orig = self.show_orig_var.get()
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for e in self.store.entries()[-limit:]:
            done = e.final
            if (show_orig or not self.settings.translate) and e.original:
                self.text.insert("end", e.original + "\n", "orig" if done else "live_orig")
            if self.settings.translate and e.translation:
                self.text.insert("end", e.translation + "\n", "trans" if e.translated_final else "live_trans")
            self.text.insert("end", "\n", "gap")
        self.text.configure(state="disabled")
        if at_bottom or self.compact_var.get():
            self.text.see("end")

    def _on_close(self) -> None:
        self.engine.stop()
        self.store.flush()
        self.root.destroy()


def run_gui(settings: Settings) -> None:
    root = tk.Tk()
    OverlayApp(root, settings)
    root.mainloop()
