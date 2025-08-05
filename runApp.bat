@echo off
echo Initializing Conda environment...

REM Check if conda is available in PATH
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda not found in PATH. Please ensure Anaconda/Miniconda is installed and added to PATH.
    pause
    exit /b 1
)

REM Navigate to the directory where this script resides
cd /d "%~dp0"

REM Activate the conda environment
echo Activating conda environment: chessEnv
call conda activate chessEnv
if %errorlevel% neq 0 (
    echo Error: Failed to activate conda environment 'chessEnv'
    echo Please ensure the environment exists: conda env list
    pause
    exit /b 1
)

REM Launch the GUI
echo Launching Chess Move Predictor...
python webGUI.py

REM Keep window open if there's an error
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)