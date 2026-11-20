@echo off
REM edge-infer 一键启动：先起 Python 推理服务，再开 Tauri 原生窗口。
REM 用法：双击本文件，或在终端执行 run_windows.bat
setlocal

set REPO=%~dp0
set PY=%REPO%python
set GUI=%REPO%tauri-app\src-tauri\target\release\edge-gui.exe

REM 设备：NPU（默认）| CPU | GPU.0
if "%EDGE_DEVICE%"=="" set EDGE_DEVICE=NPU
REM 对外 API 鉴权（留空=不鉴权）
if "%EDGE_API_KEY%"=="" set EDGE_API_KEY=
REM 模型目录：优先 8B（Agent 效果好）；未设置时由服务自动选择
if "%EDGE_MODEL_DIR%"=="" if exist "%REPO%models\qwen3-8b-int4-cw" set EDGE_MODEL_DIR=%REPO%models\qwen3-8b-int4-cw

echo [edge-infer] 启动推理服务 (device=%EDGE_DEVICE%) ...
start "edge-infer-server" cmd /c "cd /d %PY% && set EDGE_DEVICE=%EDGE_DEVICE% && set EDGE_API_KEY=%EDGE_API_KEY% && set EDGE_MODEL_DIR=%EDGE_MODEL_DIR% && python run_server.py"

echo [edge-infer] 等待服务就绪 ...
timeout /t 4 /nobreak >nul

if exist "%GUI%" (
  echo [edge-infer] 打开原生窗口 ...
  start "" "%GUI%"
) else (
  echo [edge-infer] 未找到原生窗口，改用浏览器打开 http://127.0.0.1:8080/ui/
  start "" "http://127.0.0.1:8080/ui/"
)

echo [edge-infer] 完成。API: http://127.0.0.1:8080/v1  GUI: http://127.0.0.1:8080/ui/
endlocal
