; ============================================================================
;  edge-infer Windows installer (Inno Setup 6)
; ----------------------------------------------------------------------------
;  Produces a single setup .exe that installs:
;    - edge-gui.exe              (Tauri native window)
;    - the Rust CLI helpers      (edge-detect / edge-engine / edge-tiers ...)
;    - the Python backend        (python/edge_server/**)
;    - a bundled portable Python (runtime/python/**) with all deps, so a fresh
;      machine needs NO pre-installed Python
;    - a per-user models/ dir    (weights are NOT bundled; downloaded in-app)
;    - Start-menu + Desktop shortcut
;
;  Build:  iscc installer\edge-infer.iss      (after running build_installer.ps1)
; ============================================================================

#define MyAppName        "edge-infer"
#define MyAppVersion     "0.3.1"
#define MyAppPublisher   "HelloVM"
#define MyAppURL         "https://github.com/hellovm/edge-infer"
#define MyAppExeName     "edge-gui.exe"

[Setup]
AppId={{7C3E1A52-9D44-4B2E-9F1A-3E5C8B6A0D11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Per-user install by default so no admin prompt is required.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=edge-infer-{#MyAppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
UninstallDisplayName={#MyAppName} {#MyAppVersion}
DisableDirPage=no

[Languages]
Name: "chinese"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";   Description: "创建桌面快捷方式";            GroupDescription: "快捷方式:"; Flags: checkedonce
Name: "startmenuicon"; Description: "创建开始菜单快捷方式";        GroupDescription: "快捷方式:"; Flags: checkedonce

[Files]
; ---- native exe(s) -----------------------------------------------------------
Source: "..\payload\edge-gui.exe";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\payload\edge-detect.exe";     DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\payload\edge-engine.exe";     DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\payload\edge-tiers.exe";      DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\payload\cache_bench.exe";     DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; ---- python backend ----------------------------------------------------------
Source: "..\payload\python\*";  DestDir: "{app}\python";  Flags: ignoreversion recursesubdirs createallsubdirs
; ---- bundled portable python runtime (embeddable + site-packages) ------------
Source: "..\payload\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
; ---- docs -------------------------------------------------------------------
Source: "..\payload\README.md";    DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "..\payload\LICENSE";      DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; models are downloaded in-app; create an empty dir so it exists from day one.
Name: "{app}\models"; Permissions: users-modify
Name: "{app}\logs";   Permissions: users-modify

[Icons]
Name: "{group}\{#MyAppName}";              Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}";         Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";        Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Launch the app after install (the app auto-starts the inference service).
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Leave user models/models.txt alone if they downloaded big weights? We remove
; the app tree; models/ is under {app} so it goes too — warn in the wizard.
