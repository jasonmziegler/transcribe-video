"""Download a faster-whisper model into a local folder.

Hugging Face (huggingface.co) is blocked on this network, so this script pulls
the CTranslate2 model files from ModelScope instead. Once downloaded, point
transcribe.py at the local folder:

    python download_model.py large-v3
    python transcribe.py video.mp4 --model models/faster-whisper-large-v3

Available names map to the pengzhendong/* ModelScope repos below.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ModelScope repos that hold ready-to-use CTranslate2 (faster-whisper) models.
REPOS = {
    "tiny": "pengzhendong/faster-whisper-tiny",
    "tiny.en": "pengzhendong/faster-whisper-tiny.en",
    "base": "pengzhendong/faster-whisper-base",
    "base.en": "pengzhendong/faster-whisper-base.en",
    "small": "pengzhendong/faster-whisper-small",
    "small.en": "pengzhendong/faster-whisper-small.en",
    "medium": "pengzhendong/faster-whisper-medium",
    "medium.en": "pengzhendong/faster-whisper-medium.en",
    "large-v2": "pengzhendong/faster-whisper-large-v2",
    "large-v3": "pengzhendong/faster-whisper-large-v3",
    "distil-large-v3": "pengzhendong/faster-whisper-distil-large-v3",
}

API = "https://modelscope.cn/api/v1/models/{repo}/repo"
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Files a faster-whisper model needs. The vocabulary ships as either
# vocabulary.txt (older models) or vocabulary.json (large-v3); we require
# whichever the repo actually has. Everything else present is grabbed too.
REQUIRED = ["config.json", "model.bin", "tokenizer.json"]
VOCAB = ["vocabulary.json", "vocabulary.txt"]
OPTIONAL = ["preprocessor_config.json", "vocabulary.json", "vocabulary.txt"]


def _list_files(repo: str) -> set[str]:
    url = f"{API.format(repo=repo)}/files?Revision=master&Root="
    import json

    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return {f["Path"] for f in data["Data"]["Files"]}


def _download(repo: str, filename: str, dest: Path) -> None:
    url = f"{API.format(repo=repo)}?Revision=master&FilePath={filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = read * 100 // total
                    mb = read / (1 << 20)
                    print(f"\r  {filename}: {pct:3d}% ({mb:6.1f} MiB)", end="")
    print()
    tmp.replace(dest)


def download(name: str) -> Path:
    if name not in REPOS:
        print(f"Unknown model '{name}'. Choose from: {', '.join(REPOS)}", file=sys.stderr)
        raise SystemExit(2)

    repo = REPOS[name]
    out_dir = MODELS_DIR / f"faster-whisper-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching '{name}' from ModelScope ({repo}) -> {out_dir}")
    available = _list_files(repo)

    for filename in REQUIRED:
        if filename not in available:
            print(f"Error: required file '{filename}' missing from repo.", file=sys.stderr)
            raise SystemExit(1)
        _download(repo, filename, out_dir / filename)

    if not any(v in available for v in VOCAB):
        print(f"Error: no vocabulary file ({' or '.join(VOCAB)}) in repo.", file=sys.stderr)
        raise SystemExit(1)

    for filename in OPTIONAL:
        if filename in available:
            _download(repo, filename, out_dir / filename)

    print(f"\nDone. Use it with:\n  python transcribe.py <video> --model {out_dir}")
    return out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download a faster-whisper model locally (via ModelScope).")
    p.add_argument("model", nargs="?", default="large-v3", help=f"Model name. Options: {', '.join(REPOS)}")
    args = p.parse_args(argv)
    download(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
