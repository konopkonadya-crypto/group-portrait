; Установщик для "Групповых портретов".
; Собирается на Windows командой: ISCC.exe installer.iss
; Результат: Output\GroupPortrait-Setup.exe

#define AppName "Групповые портреты"
#define AppVersion "1.0"
#define AppExe "GroupPortrait.exe"

[Setup]
AppId={{8F3A21C4-77B2-4D51-9E6A-2C1F5B8D4A73}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=SYNTX.AI
DefaultDirName={autopf}\GroupPortrait
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=GroupPortrait-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile=icon.ico

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Ярлыки:"

[Files]
Source: "dist\GroupPortrait\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Справка"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Запустить программу"; Flags: nowait postinstall skipifsilent
