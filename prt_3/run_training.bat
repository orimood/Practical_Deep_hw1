@echo off
REM Activate venv and run training script from prt_3 subdirectory
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
cd /d "%~dp0"
python train_transfer_models.py
pause
