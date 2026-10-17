; MatteDesk – Copyright (C) 2026 Lech Gruszewski
; SPDX-License-Identifier: GPL-3.0-or-later
;
; Windows installer of MatteDesk (Inno Setup 6). Do not compile by hand – run:
;     python tools/build_installer.py
; which prepares the stage folder (public copy of the code + bundled Python with the requirements + 7zr.exe) and
; passes AppVersion, StageDir, OutputDir and the ComfyUI portable size to ISCC.
;
; Per-user install (no admin rights): %LOCALAPPDATA%\Programs\MatteDesk – the app writes its config.local.json,
; logs, backups and bin\ffmpeg next to the code, so it must not live in Program Files.
; One question about ComfyUI (user decision 2026-10-17): "Do you already have ComfyUI?"
;   Yes -> detected by tools/installer_setup.py (Desktop / portable / from git), optional folder hint;
;   No  -> download the official ComfyUI portable (python_embeded) into a chosen folder.

#ifndef AppVersion
  #define AppVersion "0.0"
#endif
#ifndef StageDir
  #define StageDir "..\build\installer\stage"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
#ifndef PortableGB
  #define PortableGB "2.0"
#endif
#ifndef PortableDiskGB
  #define PortableDiskGB "6.5"
#endif

[Setup]
AppId={{6F0B7A52-3C1E-4D7B-9E55-4D61C0A8F3B2}
AppName=MatteDesk
AppVersion={#AppVersion}
AppVerName=MatteDesk {#AppVersion}
AppPublisher=Lech Gruszewski
AppPublisherURL=https://github.com/fishfinger44/MatteDesk
AppSupportURL=https://github.com/fishfinger44/MatteDesk/issues
DefaultDirName={autopf}\MatteDesk
DefaultGroupName=MatteDesk
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#StageDir}\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=MatteDesk-{#AppVersion}-setup
SetupIconFile={#StageDir}\installer\MatteDesk.ico
UninstallDisplayIcon={app}\installer\MatteDesk.ico
UninstallDisplayName=MatteDesk
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=auto
CloseApplications=yes

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "pl"; MessagesFile: "compiler:Languages\Polish.isl"

[CustomMessages]
en.ComfyTitle=ComfyUI
en.ComfyDesc=MatteDesk computes all models in ComfyUI.
en.ComfyQuestion=Where should MatteDesk get ComfyUI from?
en.ComfyYes=I already have ComfyUI
en.ComfyYesNote=MatteDesk finds it by itself (ComfyUI Desktop, portable or from git). Your models stay where they are.
en.ComfyYesDir=Folder (optional – only if it is not found):
en.ComfyNo=Download ComfyUI portable (recommended when you do not have it)
en.ComfyNoNote=The official package: {#PortableGB} GB download, about {#PortableDiskGB} GB on disk. Folder:
en.ComfyLater=Skip – I will set ComfyUI up later in the MatteDesk Doctor
en.Browse=Browse...
en.NoSpace=Not enough free space on %1 for ComfyUI portable (needs about %2 GB).
en.NeedDir=Choose the folder for ComfyUI portable.
en.SetupComfy=Setting up ComfyUI...
en.ComfyFailed=ComfyUI was not set up (see the console window). MatteDesk is installed – the Doctor will help on the first start.
en.Launch=Start MatteDesk
pl.ComfyTitle=ComfyUI
pl.ComfyDesc=MatteDesk liczy wszystkie modele w ComfyUI.
pl.ComfyQuestion=Skąd MatteDesk ma wziąć ComfyUI?
pl.ComfyYes=Mam już ComfyUI
pl.ComfyYesNote=MatteDesk sam go znajdzie (ComfyUI Desktop, portable albo z gita). Modele zostają tam, gdzie są.
pl.ComfyYesDir=Folder (opcjonalnie – tylko gdy nie zostanie znaleziony):
pl.ComfyNo=Pobierz ComfyUI portable (zalecane, gdy go nie masz)
pl.ComfyNoNote=Oficjalna paczka: {#PortableGB} GB do pobrania, ok. {#PortableDiskGB} GB na dysku. Folder:
pl.ComfyLater=Pomiń – ustawię ComfyUI później w Doktorze MatteDesk
pl.Browse=Przeglądaj...
pl.NoSpace=Za mało miejsca na %1 dla ComfyUI portable (potrzeba ok. %2 GB).
pl.NeedDir=Wybierz folder dla ComfyUI portable.
pl.SetupComfy=Konfiguracja ComfyUI...
pl.ComfyFailed=Nie udało się ustawić ComfyUI (szczegóły w oknie konsoli). MatteDesk jest zainstalowany – Doktor pomoże przy pierwszym starcie.
pl.Launch=Uruchom MatteDesk

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; start_v<version>.bat of an older version (renamed on every release)
Type: files; Name: "{app}\start_v*.bat"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\MatteDesk"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\tools\launch_app.py"""; WorkingDir: "{app}"; IconFilename: "{app}\installer\MatteDesk.ico"
Name: "{autodesktop}\MatteDesk"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\tools\launch_app.py"""; WorkingDir: "{app}"; IconFilename: "{app}\installer\MatteDesk.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\tools\launch_app.py"""; WorkingDir: "{app}"; Description: "{cm:Launch}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; everything MatteDesk wrote next to its code (config.local.json, logs, backups, bin\ffmpeg, __pycache__) and its
; browser profile; shot folders and ComfyUI (also a portable one downloaded by this installer) are never touched
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{localappdata}\MatteDesk"

[Code]
var
  ComfyPage: TWizardPage;
  RbYes, RbNo, RbLater: TNewRadioButton;
  EdYes, EdNo: TNewEdit;
  BtYes, BtNo: TNewButton;

procedure UpdateEnabled;
begin
  EdYes.Enabled := RbYes.Checked;
  BtYes.Enabled := RbYes.Checked;
  EdNo.Enabled := RbNo.Checked;
  BtNo.Enabled := RbNo.Checked;
end;

procedure RadioClick(Sender: TObject);
begin
  UpdateEnabled;
end;

procedure BrowseYes(Sender: TObject);
var
  D: String;
begin
  D := EdYes.Text;
  if BrowseForFolder(CustomMessage('ComfyYesDir'), D, False) then
    EdYes.Text := D;
end;

procedure BrowseNo(Sender: TObject);
var
  D: String;
begin
  D := EdNo.Text;
  if BrowseForFolder(CustomMessage('ComfyNo'), D, True) then
    EdNo.Text := D;
end;

function MakeRadio(Caption: String; Top: Integer): TNewRadioButton;
begin
  Result := TNewRadioButton.Create(ComfyPage);
  Result.Parent := ComfyPage.Surface;
  Result.Caption := Caption;
  Result.Top := Top;
  Result.Width := ComfyPage.SurfaceWidth;
  Result.Font.Style := [fsBold];
  Result.OnClick := @RadioClick;
end;

procedure MakeNote(Caption: String; Top: Integer);
var
  T: TNewStaticText;
begin
  T := TNewStaticText.Create(ComfyPage);
  T.Parent := ComfyPage.Surface;
  T.Left := ScaleX(18);
  T.Top := Top;
  T.Width := ComfyPage.SurfaceWidth - ScaleX(18);
  T.AutoSize := False;
  T.Height := ScaleY(28);
  T.WordWrap := True;
  T.Caption := Caption;
end;

procedure MakeDirRow(var Ed: TNewEdit; var Bt: TNewButton; Top: Integer; Default: String; OnBrowse: TNotifyEvent);
begin
  Ed := TNewEdit.Create(ComfyPage);
  Ed.Parent := ComfyPage.Surface;
  Ed.Left := ScaleX(18);
  Ed.Top := Top;
  Ed.Width := ComfyPage.SurfaceWidth - ScaleX(18) - ScaleX(90);
  Ed.Text := Default;
  Bt := TNewButton.Create(ComfyPage);
  Bt.Parent := ComfyPage.Surface;
  Bt.Caption := CustomMessage('Browse');
  Bt.Left := ComfyPage.SurfaceWidth - ScaleX(82);
  Bt.Top := Top - ScaleY(1);
  Bt.Width := ScaleX(82);
  Bt.Height := ScaleY(23);
  Bt.OnClick := OnBrowse;
end;

procedure InitializeWizard;
var
  Q: TNewStaticText;
begin
  ComfyPage := CreateCustomPage(wpSelectDir, CustomMessage('ComfyTitle'), CustomMessage('ComfyDesc'));
  Q := TNewStaticText.Create(ComfyPage);
  Q.Parent := ComfyPage.Surface;
  Q.Caption := CustomMessage('ComfyQuestion');
  // order (user decision 2026-10-17): 1. download ComfyUI portable (default), 2. I already have it, 3. later
  RbNo := MakeRadio(CustomMessage('ComfyNo'), ScaleY(22));
  MakeNote(CustomMessage('ComfyNoNote'), ScaleY(42));
  MakeDirRow(EdNo, BtNo, ScaleY(72), ExpandConstant('{sd}\ComfyUI'), @BrowseNo);
  RbYes := MakeRadio(CustomMessage('ComfyYes'), ScaleY(108));
  MakeNote(CustomMessage('ComfyYesNote'), ScaleY(128));
  MakeNote(CustomMessage('ComfyYesDir'), ScaleY(158));
  MakeDirRow(EdYes, BtYes, ScaleY(174), '', @BrowseYes);
  RbLater := MakeRadio(CustomMessage('ComfyLater'), ScaleY(210));
  // default = 1. download (user decision 2026-10-17, also when updating); a silent install (/SILENT) never starts
  // a 2 GB download by itself
  if WizardSilent then
    RbLater.Checked := True
  else
    RbNo.Checked := True;
  UpdateEnabled;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  FreeMB, TotalMB: Cardinal;
  Need: Integer;
begin
  Result := True;
  if (CurPageID = ComfyPage.ID) and RbNo.Checked then begin
    if Trim(EdNo.Text) = '' then begin
      MsgBox(CustomMessage('NeedDir'), mbError, MB_OK);
      Result := False;
      exit;
    end;
    Need := Round(({#PortableGB} + {#PortableDiskGB}) * 1024);
    if GetSpaceOnDisk(ExtractFileDrive(EdNo.Text), True, FreeMB, TotalMB) and (FreeMB < Need) then begin
      MsgBox(FmtMessage(CustomMessage('NoSpace'), [ExtractFileDrive(EdNo.Text), '{#PortableGB} + {#PortableDiskGB}']), mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Args: String;
  Code: Integer;
begin
  if CurStep <> ssPostInstall then
    exit;
  if RbYes.Checked then begin
    Args := '--comfy have';
    if Trim(EdYes.Text) <> '' then
      Args := Args + ' --dir "' + Trim(EdYes.Text) + '"';
  end else if RbNo.Checked then
    Args := '--comfy portable --dir "' + Trim(EdNo.Text) + '"'
  else
    exit;
  WizardForm.StatusLabel.Caption := CustomMessage('SetupComfy');
  if not Exec(ExpandConstant('{app}\python\python.exe'), '"' + ExpandConstant('{app}\tools\installer_setup.py') + '" ' + Args,
              ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, Code) or (Code <> 0) then
    MsgBox(CustomMessage('ComfyFailed'), mbInformation, MB_OK);
end;
