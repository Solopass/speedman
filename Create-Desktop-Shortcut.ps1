# ==============================================================================
# Create Desktop and Explorer SendTo Shortcuts for SOL Speedman
# ==============================================================================
$ErrorActionPreference = "Stop"

$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Speedman.lnk"
$targetScript = "D:\Workspace\speedman\Speedman-Silent.vbs"
$workingDir = "D:\Workspace\speedman"

$wscriptPath = Join-Path $env:SystemRoot "System32\wscript.exe"

$wshShell = New-Object -ComObject WScript.Shell

# 1. Desktop Shortcut
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $wscriptPath
$shortcut.Arguments = "`"$targetScript`""
$shortcut.WorkingDirectory = $workingDir
$shortcut.Description = "Speedman - Ultra-Speed Speech Engine (Port 8081)"
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,168"
$shortcut.Save()

Write-Host "Created Desktop Shortcut: $shortcutPath" -ForegroundColor Green

# 2. Explorer "Send to" Shortcut
$sendToPath = Join-Path $env:APPDATA "Microsoft\Windows\SendTo"
if (Test-Path $sendToPath) {
    # 2a. SendTo: 5x Fast
    $sendToShortcutPath = Join-Path $sendToPath "Speedman (5x Fast).lnk"
    $sendToShortcut = $wshShell.CreateShortcut($sendToShortcutPath)
    $sendToShortcut.TargetPath = "D:\Workspace\speedman\speedman-sendto.bat"
    $sendToShortcut.WorkingDirectory = $workingDir
    $sendToShortcut.Description = "Compress speech audio 5x with Speedman Fast preset"
    $sendToShortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,168"
    $sendToShortcut.Save()
    Write-Host "Created Explorer SendTo Shortcut: $sendToShortcutPath" -ForegroundColor Green

    # 2b. SendTo: 5x + Transcribe (Parakeet ASR)
    $sendToTrShortcutPath = Join-Path $sendToPath "Speedman (5x + Transcribe).lnk"
    $sendToTrShortcut = $wshShell.CreateShortcut($sendToTrShortcutPath)
    $sendToTrShortcut.TargetPath = "D:\Workspace\speedman\speedman-sendto-transcribe.bat"
    $sendToTrShortcut.WorkingDirectory = $workingDir
    $sendToTrShortcut.Description = "Compress 5x and sync Parakeet ASR subtitles (.vtt)"
    $sendToTrShortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,138"
    $sendToTrShortcut.Save()
    Write-Host "Created Explorer SendTo Shortcut: $sendToTrShortcutPath" -ForegroundColor Green

    # 2c. SendTo: 5x + Note Summary (Obsidian 1Notebook)
    $sendToObsShortcutPath = Join-Path $sendToPath "Speedman (5x + Note Summary).lnk"
    $sendToObsShortcut = $wshShell.CreateShortcut($sendToObsShortcutPath)
    $sendToObsShortcut.TargetPath = "D:\Workspace\speedman\speedman-sendto-obsidian.bat"
    $sendToObsShortcut.WorkingDirectory = $workingDir
    $sendToObsShortcut.Description = "Compress 5x, transcribe, and export Obsidian summary note"
    $sendToObsShortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,266"
    $sendToObsShortcut.Save()
    Write-Host "Created Explorer SendTo Shortcut: $sendToObsShortcutPath" -ForegroundColor Green
}
