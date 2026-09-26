# transcribe-video

Transcribe video (or audio) files to text using a **local** Whisper model.
Runs fully offline after the model is downloaded, and uses your NVIDIA GPU
(RTX 3060) automatically, falling back to CPU if needed.

No cloud, no API keys — the audio never leaves your machine.

## What's here

| File | Purpose |
|------|---------|
| `transcribe.py`     | The transcriber CLI. |
| `download_model.py` | Downloads Whisper models into `./models`. |
| `transcribe.bat`    | Convenience wrapper (no venv activation needed). |
| `requirements.txt`  | Python dependencies. |
| `models/`           | Downloaded models live here. |

## One-time setup

Already done on this machine, but for reference:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Note about model downloads

`huggingface.co` (Whisper's usual model host) is **blocked on this network**,
so `download_model.py` fetches the models from **ModelScope** instead and stores
them locally in `./models`. `transcribe.py` automatically finds a model there
when you pass a bare name like `large-v3`.

## Usage

Download a model once (default is `large-v3`, the most accurate):

```powershell
.\transcribe.bat --help                 REM show all options
.venv\Scripts\python.exe download_model.py large-v3
```

Then transcribe a file:

```powershell
.\transcribe.bat "C:\Users\jason\Videos\meeting.mp4"
```

This writes `meeting.txt` and `meeting.srt` next to the video.

### Common options

```powershell
REM Choose output formats (txt, srt, vtt)
.\transcribe.bat video.mp4 --formats txt srt vtt

REM Force a language (skips auto-detect, a bit faster/more reliable)
.\transcribe.bat video.mp4 --language en

REM Use a smaller/faster model (download it first)
.venv\Scripts\python.exe download_model.py small
.\transcribe.bat video.mp4 --model small

REM Write outputs to a specific folder
.\transcribe.bat video.mp4 --output-dir "C:\transcripts"

REM Force CPU (or force GPU)
.\transcribe.bat video.mp4 --device cpu
```

## Models (quality vs. speed)

| Model            | Rough size | Notes |
|------------------|-----------|-------|
| `tiny`           | ~75 MB    | Fastest, least accurate. Good for quick tests. |
| `base` / `small` | ~150–500 MB | Decent, fast. |
| `medium`         | ~1.5 GB   | Good balance. |
| `large-v3`       | ~3 GB     | **Most accurate** (default). Fits your 6 GB GPU. |
| `distil-large-v3`| ~1.5 GB   | Near large-v3 quality, ~2x faster. |

English-only variants (`tiny.en`, `base.en`, ...) are slightly better for
English audio.

## How it works

- **faster-whisper** (CTranslate2) runs the Whisper model on the GPU.
- **PyAV** (bundled with faster-whisper) decodes the video's audio directly, so
  no separate ffmpeg install is required — most video/audio containers work.
- CUDA/cuDNN runtime DLLs come from the `nvidia-*-cu12` pip wheels; the script
  preloads them at startup so no manual CUDA install is needed.
