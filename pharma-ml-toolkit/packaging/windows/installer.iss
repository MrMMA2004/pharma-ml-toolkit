; PharmaML Studio — the offline Windows installer (part9 §2/§7/§11).
;
; Architecture: PyInstaller onedir dist (release\windows\dist\PharmaMLStudio)
; packaged by Inno Setup into ONE self-contained Setup EXE.  The payload
; embeds the Python runtime, PySide6/Qt, RDKit, torch-CPU and the whole
; scientific stack, so the target machine needs NOTHING installed
; except Windows 10/11 x64 (the VC++ runtime DLLs the stack needs are
; bundled app-local by PyInstaller; see README.md §VC runtime).
;
; MyAppVersion is passed on the ISCC command line by build_windows.ps1
; (/DMyAppVersion=x.y.z — read from the toolkit VERSION file), so this
; script itself never hardcodes a stale version.
;
; Data-architecture contract (part9 §7):
;   * installation files -> {app} (Program Files; immutable)
;   * user config         -> HKCU\Software\PharmaML (QSettings — the
;                            app's AppSettings uses organization
;                            "PharmaML", application "PharmaML Studio")
;   * user data           -> {userappdata}\PharmaML\PharmaML Studio and
;                            the user's Documents (datasets, exports,
;                            saved models) — chosen by QStandardPaths
;                            inside the app.
;   * uninstall NEVER silently deletes user data: the explicit
;     opt-in question below is the ONLY removal path (part9 §7 rule).

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-no-version-passed"
#endif

#define MyAppName "PharmaML Studio"
#define MyAppPublisher "PharmaML contributors"
#define MyAppExeName "PharmaMLStudio.exe"
#define MyAppId "{{6F7A5B3C-9E21-4B74-8C0D-1A2B3C4D5E6F}"
; A stable, hand-assigned AppId: upgrades replace, never duplicate.

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/MrMMA2004/pharma-ml-toolkit
AppUpdatesURL=https://github.com/MrMMA2004/pharma-ml-toolkit/releases
DefaultDirName={autopf}\PharmaML Studio
DefaultGroupName=PharmaML Studio
DisableProgramGroupPage=yes
; The MIT license (the toolkit LICENSE file) is displayed at install:
LicenseFile=..\..\LICENSE
OutputDir=..\..\release\windows
OutputBaseFilename=PharmaML_Studio_Setup_v{#MyAppVersion}
SetupIconFile=assets\pharmaml.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
; Professional honesty (part9 §11): unsigned build, real metadata,
; spaces and non-ASCII install paths are supported by the {app} macros.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The complete PyInstaller onedir tree (Python + Qt + science + app):
Source: "..\..\release\windows\dist\PharmaMLStudio\*"; \
    DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; The redistribution-license inventory travels inside the install
; (part9 §15: include applicable license files and notices):
Source: "..\..\release\windows\THIRD_PARTY_LICENSES\*"; \
    DestDir: "{app}\THIRD_PARTY_LICENSES"; \
    Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\PharmaML Studio"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall PharmaML Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\PharmaML Studio"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,PharmaML Studio}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; INTENTIONALLY EMPTY — part9 §7: uninstallation must never silently
; delete user-generated datasets/models/reports.  The ONLY removal
; path is the explicit opt-in question in the code section below.

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataRoot: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    DataRoot := ExpandConstant('{userappdata}') + '\PharmaML';
    if MsgBox(
      'Also remove user data?' + #13#10#13#10 +
      'This deletes your saved settings, datasets, exported figures, ' +
      'reports and trained models under:' + #13#10 + DataRoot + #13#10 +
      'and the registry settings under HKCU\Software\PharmaML.' + #13#10#13#10 +
      'Choose No to keep them (the default).',
      mbConfirmation, MB_YESNO) = IDYES then
    begin
      DelTree(DataRoot, True, True, True);
      RegDeleteKeyIncludingSubkeys(HKCU, 'Software\PharmaML');
    end;
  end;
end;
