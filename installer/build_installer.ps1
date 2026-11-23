# ============================================================================
#  build_installer.ps1 — assemble payload + build the edge-infer setup .exe
# ----------------------------------------------------------------------------
#  Steps:
#    1. cargo build --release  (Rust CLI helpers)
#    2. cargo build --release  (Tauri GUI  -> edge-gui.exe)
#    3. download Python *embeddable* zip, unpack into payload\runtime\python
#    4. pip install the backend deps *into* that runtime (Lib\site-packages)
#    5. copy python/ backend + exes + docs into payload\
#    6. run Inno Setup (iscc) to produce dist\edge-infer-<ver>-setup.exe
#
#  Usage:  powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1
#  Requires: cargo (Rust), internet (for the embeddable python + pip wheels).
#            Inno Setup 6 (iscc.exe on PATH or in Program Files).
# ============================================================================
$ErrorActionPreference = "Stop"

$Root    = Split-Path -Parent $PSScriptRoot          # repo root
$PyVer   = "3.11.9"                                   # embeddable runtime version
$PyTag   = "311"                                      # -> python3xx._pth / dll name
$Payload = Join-Path $Root "installer\payload"
$Runtime = Join-Path $Payload "runtime\python"
$Dist    = Join-Path $Root "dist"

function Step($m) { Write-Host "=== $m ===" -ForegroundColor Cyan }
function Ensure-Dir($p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }

# --- 0. clean payload --------------------------------------------------------
Step "Preparing payload dir"
if (Test-Path $Payload) { Remove-Item -Recurse -Force $Payload }
Ensure-Dir $Payload
Ensure-Dir $Dist

# --- 1. Rust CLI helpers -----------------------------------------------------
Step "Building Rust CLI helpers (release)"
Push-Location $Root
& cargo build --release
Pop-Location

# --- 2. Tauri GUI ------------------------------------------------------------
Step "Building Tauri GUI (release)"
$SdkEnv = Join-Path $Root "scripts\env.ps1"
if (Test-Path $SdkEnv) { . $SdkEnv }   # may export PATH for windres etc.
Push-Location (Join-Path $Root "tauri-app\src-tauri")
& cargo build --release
Pop-Location

# --- 3. embeddable Python ----------------------------------------------------
Step "Fetching Python $PyVer (embeddable)"
$pyZip = Join-Path $env:TEMP "python-$PyVer-embed-amd64.zip"
if (-not (Test-Path $pyZip)) {
  $url = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip"
  Invoke-WebRequest -Uri $url -OutFile $pyZip
}
Ensure-Dir $Runtime
Expand-Archive -Path $pyZip -DestinationPath $Runtime -Force

# Enable site-packages: an embeddable distro ships python3xx._pth which lists
# only the zip + "."; uncomment "import site" and add Lib\site-packages so pip
# and our deps are importable.
$pth = Join-Path $Runtime "python$PyTag._pth"
if (Test-Path $pth) {
  $lines = Get-Content $pth | ForEach-Object {
    if ($_ -match '^\s*#\s*import\s+site') { 'import site' } else { $_ }
  }
  if (-not ($lines -match '^\s*import site\s*$')) { $lines += 'import site' }
  if (-not ($lines -match 'Lib\\site-packages')) { $lines += 'Lib\site-packages' }
  Set-Content -Path $pth -Value $lines -Encoding ASCII
}

# --- 4. install backend deps into the runtime --------------------------------
Step "Installing backend dependencies into bundled runtime"
$py = Join-Path $Runtime "python.exe"
# bootstrap pip
$getpip = Join-Path $env:TEMP "get-pip.py"
if (-not (Test-Path $getpip)) { Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip }
& $py $getpip --no-warn-script-location
$req = Join-Path $Root "python\requirements.txt"
& $py -m pip install --no-warn-script-location --upgrade -r $req

# --- 5. stage backend + exes + docs -----------------------------------------
Step "Staging files into payload"
Copy-Item (Join-Path $Root "tauri-app\src-tauri\target\release\edge-gui.exe") $Payload -Force
foreach ($cli in @("edge-detect.exe","edge-engine.exe","edge-tiers.exe","cache_bench.exe")) {
  $src = Join-Path $Root "target\release\$cli"
  if (Test-Path $src) { Copy-Item $src $Payload -Force }
}
Copy-Item (Join-Path $Root "python") (Join-Path $Payload "python") -Recurse -Force
# drop dev-only caches from the staged backend
Get-ChildItem (Join-Path $Payload "python") -Recurse -Directory -Include "__pycache__",".venv" |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Root "README.md") $Payload -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Root "LICENSE")   $Payload -Force -ErrorAction SilentlyContinue

# --- 6. compile the installer ------------------------------------------------
Step "Running Inno Setup compiler"
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
  foreach ($p in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                   "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
    if (Test-Path $p) { $iscc = $p; break }
  }
}
if (-not $iscc) {
  Write-Warning "未找到 iscc.exe（Inno Setup 6）。payload 已生成，可在安装 Inno Setup 后手动运行："
  Write-Warning "  iscc `"$Root\installer\edge-infer.iss`""
  exit 0
}
& $iscc (Join-Path $Root "installer\edge-infer.iss")
Step "Done — 安装包在 $Dist"
Get-ChildItem $Dist -Filter "*.exe" | ForEach-Object { Write-Host "  -> $($_.FullName)" }
