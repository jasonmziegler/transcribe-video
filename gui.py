"""A simple desktop GUI for the local video transcriber.

Select a video/audio file, pick an output folder, choose a model and formats,
then watch the transcript stream in with a real progress bar. Transcription runs
on a background thread so the window stays responsive.

Run it with:   pythonw gui.py     (or double-click gui.bat)
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

# Reuse the tested logic from the CLI.
import transcribe

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


def downloaded_models() -> list[str]:
    """Names of models already downloaded into ./models (bare name only)."""
    if not MODELS_DIR.is_dir():
        return []
    names = []
    for folder in sorted(MODELS_DIR.glob("faster-whisper-*")):
        if (folder / "model.bin").is_file():
            names.append(folder.name.removeprefix("faster-whisper-"))
    return names


class TranscriberGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Local Video Transcriber")
        root.minsize(680, 520)

        # Messages flow from the worker thread to the UI through this queue.
        self.msgs: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()

        self._build_widgets()
        self._poll_queue()

    # ---------- layout ----------
    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        # Input file
        ttk.Label(frm, text="Video / audio file:").grid(row=0, column=0, sticky="w", **pad)
        self.input_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse…", command=self._pick_input).grid(row=0, column=2, **pad)

        # Output folder
        ttk.Label(frm, text="Output folder:").grid(row=1, column=0, sticky="w", **pad)
        self.output_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse…", command=self._pick_output).grid(row=1, column=2, **pad)

        # Options row: model, language, device
        opts = ttk.Frame(frm)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Label(opts, text="Model:").pack(side="left")
        models = downloaded_models() or ["large-v3"]
        default_model = "large-v3" if "large-v3" in models else models[0]
        self.model_var = tk.StringVar(value=default_model)
        ttk.Combobox(
            opts, textvariable=self.model_var, values=models, width=18, state="readonly"
        ).pack(side="left", padx=(4, 16))

        ttk.Label(opts, text="Language:").pack(side="left")
        self.lang_var = tk.StringVar(value="en")
        ttk.Entry(opts, textvariable=self.lang_var, width=8).pack(side="left", padx=(4, 16))

        ttk.Label(opts, text="Device:").pack(side="left")
        self.device_var = tk.StringVar(value="auto")
        ttk.Combobox(
            opts, textvariable=self.device_var, values=["auto", "cuda", "cpu"],
            width=7, state="readonly",
        ).pack(side="left", padx=(4, 16))

        # Formats
        ttk.Label(opts, text="Formats:").pack(side="left")
        self.fmt_vars = {"txt": tk.BooleanVar(value=True),
                         "srt": tk.BooleanVar(value=True),
                         "vtt": tk.BooleanVar(value=False)}
        for name, var in self.fmt_vars.items():
            ttk.Checkbutton(opts, text=name, variable=var).pack(side="left")

        # Action buttons
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        self.start_btn = ttk.Button(btns, text="Transcribe", command=self._start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)

        # Progress bar + status
        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self.status_var).grid(
            row=5, column=0, columnspan=3, sticky="w", **pad
        )

        # Live transcript log
        frm.rowconfigure(6, weight=1)
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    # ---------- file pickers ----------
    def _pick_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video or audio file",
            filetypes=[
                ("Media files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4a *.mp3 *.wav *.flac *.aac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(str(Path(path).parent))

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select an output folder")
        if path:
            self.output_var.set(path)

    # ---------- run control ----------
    def _start(self) -> None:
        src = self.input_var.get().strip()
        if not src or not Path(src).is_file():
            self._set_status("Please choose a valid input file.")
            return
        formats = [f for f, v in self.fmt_vars.items() if v.get()]
        if not formats:
            self._set_status("Pick at least one output format.")
            return
        out_dir = self.output_var.get().strip() or str(Path(src).parent)

        self._clear_log()
        self.progress["value"] = 0
        self.cancel_event.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        args = dict(
            src=src,
            out_dir=out_dir,
            model=self.model_var.get(),
            language=self.lang_var.get().strip() or None,
            device=self.device_var.get(),
            formats=formats,
        )
        self.worker = threading.Thread(target=self._run_transcription, kwargs=args, daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self._set_status("Cancelling…")

    # ---------- worker thread ----------
    def _run_transcription(self, src, out_dir, model, language, device, formats) -> None:
        """Runs off the UI thread. Communicates only via self.msgs."""
        try:
            self.msgs.put(("status", "Loading model… (first run of a session takes a few seconds)"))
            transcribe._enable_bundled_cuda()
            from faster_whisper import WhisperModel

            model_ref = transcribe.resolve_model(model)
            dev, compute = transcribe.pick_device(device)
            try:
                wm = WhisperModel(model_ref, device=dev, compute_type=compute)
            except Exception as exc:
                if dev == "cuda":
                    self.msgs.put(("log", f"[GPU load failed: {exc}\n falling back to CPU]\n"))
                    dev, compute = "cpu", "int8"
                    wm = WhisperModel(model_ref, device=dev, compute_type=compute)
                else:
                    raise

            self.msgs.put(("status", f"Transcribing on {dev}…"))
            segments, info = wm.transcribe(
                src, language=language, vad_filter=True, beam_size=5
            )
            duration = info.duration or 0
            self.msgs.put((
                "log",
                f"[Detected language: {info.language} "
                f"({info.language_probability:.0%}), duration {duration:.0f}s]\n\n",
            ))

            collected = []
            for seg in segments:
                if self.cancel_event.is_set():
                    self.msgs.put(("cancelled", None))
                    return
                collected.append(seg)
                stamp = transcribe._format_timestamp(seg.start, ".")[:-4]
                self.msgs.put(("log", f"[{stamp}] {seg.text.strip()}\n"))
                if duration:
                    self.msgs.put(("progress", min(100.0, seg.end / duration * 100)))

            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            written = []
            stem = Path(src).stem
            for fmt in formats:
                dest = out_path / f"{stem}.{fmt}"
                transcribe.WRITERS[fmt](collected, dest)
                written.append(str(dest))

            self.msgs.put(("done", written))
        except Exception:
            self.msgs.put(("error", traceback.format_exc()))

    # ---------- UI-thread queue pump ----------
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "done":
                    self.progress["value"] = 100
                    self._set_status("Done. Wrote: " + ", ".join(payload))
                    self._append_log("\n[Finished]\n" + "\n".join(payload) + "\n")
                    self._reset_buttons()
                elif kind == "cancelled":
                    self._set_status("Cancelled.")
                    self._reset_buttons()
                elif kind == "error":
                    self._set_status("Error — see log below.")
                    self._append_log("\n[ERROR]\n" + payload + "\n")
                    self._reset_buttons()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ---------- small UI helpers ----------
    def _reset_buttons(self) -> None:
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    TranscriberGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
