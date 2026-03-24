@echo off
echo [v-code] Installing...

:: Download v-code.exe
echo [v-code] Downloading v-code.exe...
curl -sLO https://github.com/LeeSangMin1029/hnsw-model2vec/releases/latest/download/v-code.exe
if not exist v-code.exe (
    echo [v-code] Download failed.
    exit /b 1
)

:: Copy to PATH
if not exist "%USERPROFILE%\.cargo\bin" mkdir "%USERPROFILE%\.cargo\bin"
copy /Y v-code.exe "%USERPROFILE%\.cargo\bin\v-code.exe" >nul
del v-code.exe
echo [v-code] Installed to %USERPROFILE%\.cargo\bin\v-code.exe

:: Install nightly
echo [v-code] Installing nightly rustc...
rustup toolchain install nightly
rustup component add rust-src rustc-dev llvm-tools-preview --toolchain nightly
if errorlevel 1 (
    echo [v-code] rustup not found. Install Rust first: https://rustup.rs
    exit /b 1
)

echo [v-code] Done! Run: v-code add .code.db .
