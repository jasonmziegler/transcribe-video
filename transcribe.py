"""Transcribe a video (or audio) file to text using a local Whisper model.

Runs fully offline after the first model download. Uses the NVIDIA GPU when
available and falls back to CPU automatically.

Usage:
    python transcribe.py "C:\\path\\to\\video.mp4"
    python transcribe.py video.mp4 --model large-v3 --formats txt srt vtt
    python transcribe.py video.mp4 --language en --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _enable_bundled_cuda() -> None:
    """Make the pip-installed NVIDIA CUDA/cuDNN DLLs loadable at runtime.

    faster-whisper's GPU backend (ctranslate2) loads cuBLAS/cuDNN DLLs by bare
    name (e.g. LoadLibrary("cublas64_12.dll")), which does NOT honor
    add_dll_directory under Python's DLL search flags. So we do two things:

      1. Register the wheels' bin dirs with add_dll_directory (for dependency
         resolution of the DLLs we load), and
      2. Preload the DLLs into the process with ctypes. Once a module is loaded,
         a later LoadLibrary by the same base name resolves to it without any
         path search — which is what ctranslate2 relies on.

    No-op if the `nvidia-*-cu12` wheels aren't installed (e.g. CPU-only setup).
    """
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return

    import ctypes

    bin_dirs = []
    for root in nvidia.__path__:
        base = Path(root)
        for dll_dir in list(base.glob("*/bin")) + list(base.glob("*/lib")):
            if dll_dir.is_dir():
                bin_dirs.append(dll_dir)
                try:
                    os.add_dll_directory(str(dll_dir))
                except OSError:
                    pass

    # Preload every DLL, making several passes so dependency ordering
    # (e.g. cublas depends on cublasLt) resolves itself.
    pending = [p for d in bin_dirs for p in d.glob("*.dll")]
    for _ in range(len(pending) + 1):
        if not pending:
            break
        still = []
        for dll in pending:
            try:
                ctypes.WinDLL(str(dll))
            except OSError:
                still.append(dll)
        if len(still) == len(pending):
            break  # no progress; give up on the rest
        pending = still


def _format_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    ms = round(seconds * 1000.0)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def write_txt(segments, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def write_srt(segments, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{_format_timestamp(seg.start, ',')} --> "
                f"{_format_timestamp(seg.end, ',')}\n"
            )
            f.write(seg.text.strip() + "\n\n")


def write_vtt(segments, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(
                f"{_format_timestamp(seg.start, '.')} --> "
                f"{_format_timestamp(seg.end, '.')}\n"
            )
            f.write(seg.text.strip() + "\n\n")


WRITERS = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe a video/audio file with a local Whisper model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to the video or audio file.")
    p.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (tiny, base, small, medium, large-v3, "
        "distil-large-v3, ...). Downloaded and cached on first use.",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["txt", "srt"],
        choices=sorted(WRITERS),
        help="Output formats to write next to the input file.",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. en, es). Omit to auto-detect.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device. 'auto' uses the GPU if available.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Defaults to the input file's folder.",
    )
    return p.parse_args(argv)


def resolve_model(name: str) -> str:
    """Turn a bare model name into a local folder path when we have it cached.

    huggingface.co is blocked on this network, so models are pre-downloaded to
    ./models via download_model.py. If a matching local folder exists, use it;
    otherwise pass the name through (a full path, or a name faster-whisper will
    try to fetch from the Hub).
    """
    if Path(name).is_dir():
        return name
    local = Path(__file__).resolve().parent / "models" / f"faster-whisper-{name}"
    if (local / "model.bin").is_file():
        return str(local)
    return name


def pick_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type)."""
    if requested in ("auto", "cuda"):
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        if requested == "cuda":
            print("Requested CUDA but no GPU is available; using CPU.", file=sys.stderr)
    return "cpu", "int8"


def main(argv=None) -> int:
    args = parse_args(argv)

    src = Path(args.input).expanduser()
    if not src.is_file():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    _enable_bundled_cuda()
    from faster_whisper import WhisperModel

    model_ref = resolve_model(args.model)

    device, compute_type = pick_device(args.device)
    print(f"Loading model '{model_ref}' on {device} ({compute_type})...")
    try:
        model = WhisperModel(model_ref, device=device, compute_type=compute_type)
    except Exception as exc:
        if device == "cuda":
            print(f"GPU load failed ({exc}); falling back to CPU.", file=sys.stderr)
            device, compute_type = "cpu", "int8"
            model = WhisperModel(model_ref, device=device, compute_type=compute_type)
        else:
            raise

    print(f"Transcribing: {src.name}")
    start = time.time()
    segments, info = model.transcribe(
        str(src),
        language=args.language,
        vad_filter=True,
        beam_size=5,
    )
    print(
        f"Detected language: {info.language} "
        f"(probability {info.language_probability:.2f}); "
        f"audio duration {info.duration:.0f}s"
    )

    # Stream segments to the console and collect them for file output.
    collected = []
    for seg in segments:
        collected.append(seg)
        stamp = _format_timestamp(seg.start, ".")[:-4]
        print(f"[{stamp}] {seg.text.strip()}")

    stem = src.stem
    written = []
    for fmt in args.formats:
        out_path = out_dir / f"{stem}.{fmt}"
        WRITERS[fmt](collected, out_path)
        written.append(out_path)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
