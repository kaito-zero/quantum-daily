@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
cmd /k ".venv\Scripts\activate.bat"