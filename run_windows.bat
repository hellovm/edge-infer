@echo off
REM edge-infer 启动器（备用方式）。
REM 正常使用：直接双击 tauri-app\src-tauri\target\release\edge-gui.exe
REM —— 该应用会自动拉起 Python 推理服务，无需本脚本。
REM
REM 本脚本用于：没有编译原生窗口、或想用浏览器打开时。

setlocal
set REPO=%~dp0

REM 首选：原生窗口（会自动拉起服务）
set GUI=%REPO%tauri-app\src-tauri\target\release\edge-gui.exe
if exist "%GUI%" (
  echo [edge-infer] 启动原生窗口（服务将自动拉起）...
  start "" "%GUI%"
  goto :eof
)

REM 回退：手动起服务 + 浏览器
echo [edge-infer] 未找到原生窗口，改为「服务 + 浏览器」模式。
set PY=%REPO%python
if "%EDGE_DEVICE%"=="" set EDGE_DEVICE=AUTO
if "%EDGE_MODEL_DIR%"=="" if exist "%REPO%models\qwen3-8b-int4-cw" set EDGE_MODEL_DIR=%REPO%models\qwen3-8b-int4-cw

start "edge-infer-server" cmd /c "cd /d %PY% && set EDGE_DEVICE=%EDGE_DEVICE% && set EDGE_MODEL_DIR=%EDGE_MODEL_DIR% && python run_server.py"
echo [edge-infer] 等待服务就绪 ...
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:8080/ui/"
echo [edge-infer] 完成。API: http://127.0.0.1:8080/v1
endlocal
