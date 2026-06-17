@echo off
rem ===========================================================================
rem  Spring Embedder Control Panel launcher  (Windows)
rem  Double-click this file to build (first time only) and open the UI.
rem ===========================================================================
cd /d "%~dp0"

rem -- Make the MSYS2 / UCRT64 toolchain available if it is installed.
if exist "C:\msys64\ucrt64\bin" set "PATH=C:\msys64\ucrt64\bin;%PATH%"

rem -- Build the C++ layout engine the first time, if it isn't there yet.
if not exist "build\fr_batch.exe" (
    echo Building fr_batch for the first time...
    if not exist "build\CMakeCache.txt" cmake -S . -B build -G Ninja
    cmake --build build --target fr_batch
    if errorlevel 1 (
        echo.
        echo Build failed. See README.md for build instructions.
        pause
        exit /b 1
    )
)

rem -- Launch the UI (pythonw = no extra console window).
where pythonw >nul 2>nul && (pythonw ui.py) || (python ui.py)

if errorlevel 1 (
    echo.
    echo The UI failed to start. Run "python ui.py" in a terminal to see why.
    echo You may need:  pip install matplotlib pandas
    pause
)
