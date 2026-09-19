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
    $sendToShortcutPath = Join-Path $sendToPath "Speedman (5x Fast).lnk"
    $sendToShortcut = $wshShell.CreateShortcut($sendToShortcutPath)
    $sendToShortcut.TargetPath = "D:\Workspace\speedman\speedman-sendto.bat"
    $sendToShortcut.WorkingDirectory = $workingDir
    $sendToShortcut.Description = "Compress speech audio 5x with Speedman Fast preset"
    $sendToShortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,168"
    $sendToShortcut.Save()
    Write-Host "Created Explorer SendTo Shortcut: $sendToShortcutPath" -ForegroundColor Green
}
