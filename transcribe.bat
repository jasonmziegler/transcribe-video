@echo off
REM Convenience wrapper: run the transcriber using the project's venv Python.
REM Usage: transcribe "C:\path\to\video.mp4" [extra options]
"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe.py" %*
