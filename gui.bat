@echo off
REM Launch the transcriber GUI (no console window) using the project's venv.
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0gui.py"
