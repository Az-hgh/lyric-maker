; -*- coding: utf-8 -*-
;
; lyric-maker 安装包构建脚本（Inno Setup 6）
;
; 设计要点：
;   * 安装到 %LOCALAPPDATA%\lyric-maker（当前用户，无需管理员权限）。
;     —— 因为软件会把模型和 settings.json 写在「exe 旁边」，而 Program Files
;        对普通用户是只读的。放 AppData 才能保证安装后还能下载模型、保存设置。
;   * 安装包【不含】语音识别模型（large-v3 单独 2.9 GB）。用户装完第一次打开
;     「本地模型」页面，从 small / medium / large-v3 等里任选下载。
;   * 卸载时默认【保留】已下载的模型（在 AppData 里），只删程序本体；
;     用 /MODELS=delete 才连带删模型。
;
; 用法：用 Inno Setup 的 ISCC.exe 编译本文件即可：
;     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" lyric-maker.iss
; 产物：dist_pkg\lyric-maker-setup.exe

#define MyAppName "lyric-maker"
#define MyAppVer "1.0.0"
#define MyAppPublisher "lyric-maker"
#define MyAppURL "https://github.com/Az-hgh/lyric-maker"
#define MySource "App"

[Setup]
AppId={{lyric-maker-2026-audio2lrc}}
AppName={#MyAppName}
AppVersion={#MyAppVer}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
; 每用户安装：不需要管理员，模型/设置才能写在 exe 旁边
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UninstallDisplayIcon={app}\lyric-maker.exe
OutputDir=.
OutputBaseFilename={#MyAppName}-setup
SetupIconFile=
Compression=lzma2/ultra64
SolidCompression=yes
; 573 MB 的素材，给足内存让 lzma2 跑得快些
LZMAUseSeparateProcess=yes
InternalCompressLevel=ultra
WizardResizable=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
; 不写注册表里的「所有用户」位置
UsePreviousAppDir=yes

[Files]
; 整个 App 目录（已排除语音模型）原样装进 {app}
Source: "{#MySource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\lyric-maker.exe"; WorkingDir: "{app}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\lyric-maker.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "额外快捷方式:"

[Run]
; 装完直接打开，用户可在「本地模型」页选模型下载
Filename: "{app}\lyric-maker.exe"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 默认不动模型；带 /MODELS=delete 才删
Type: filesandordirs; Name: "{app}\models"; Check: DeleteModels

[Code]
var
  FDeleteModels: Boolean;

function InitializeSetup(): Boolean;
begin
  FDeleteModels := False;
  if ExpandConstant('{param:MODELS}') = 'delete' then
    FDeleteModels := True;
  Result := True;
end;

function DeleteModels(): Boolean;
begin
  Result := FDeleteModels;
end;
