@echo off

REM ============================================================
REM DEVSIM environment activation
REM ============================================================

set "VENV_DIR=C:\devsim-work\denv"
set "PROJECT_DIR=C:\devsim-work\my_devices\mos2_gaa_flash"
set "MKL_BIN=%VENV_DIR%\Library\bin"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment activation file not found:
    echo %VENV_DIR%\Scripts\activate.bat
    pause
    exit /b 1
)

if not exist "%MKL_BIN%\mkl_rt.3.dll" (
    echo.
    echo [ERROR] MKL runtime DLL not found:
    echo %MKL_BIN%\mkl_rt.3.dll
    pause
    exit /b 1
)

REM Activate virtual environment.
call "%VENV_DIR%\Scripts\activate.bat"

REM Add Intel MKL runtime directory.
set "PATH=%MKL_BIN%;%PATH%"

REM Tell DEVSIM which MKL runtime to load.
set "DEVSIM_MATH_LIBS=mkl_rt.3.dll"

REM Move to project directory.
cd /d "%PROJECT_DIR%"

echo.
echo ============================================================
echo DEVSIM ENVIRONMENT ACTIVATED
echo ============================================================

echo.
echo Python executable:
where python

echo.
echo Current Python:
python -c "import sys; print(sys.executable)"

echo.
echo DEVSIM math library:
echo %DEVSIM_MATH_LIBS%

echo.
echo Testing DEVSIM import...
python -c "import devsim; print('DEVSIM import successful')"

if errorlevel 1 (
    echo.
    echo [ERROR] DEVSIM import failed.
    pause
    exit /b 1
)

echo.
echo Environment setup completed.
echo.
echo Run the following command:
echo python 05_program_erase\run_coarse_voltage_sweep.py
echo.

cmd /k