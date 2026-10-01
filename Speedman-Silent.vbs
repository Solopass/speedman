' ==============================================================================
' SOL Speedman - Silent Windows Launcher
' Launches Speedman Speech Studio with ZERO popup console windows.
' ==============================================================================
Option Explicit

Dim objShell, strWslDistro, strUrl, objHttp, isAlive, i, objFso, chromePath, bravePath

Set objShell = CreateObject("WScript.Shell")
Set objFso = CreateObject("Scripting.FileSystemObject")
strWslDistro = "Ubuntu-24.04"
strUrl = "http://127.0.0.1:8081/"

' 1. Keep WSL background session active so Windows does not auto-terminate the VM
objShell.Run "wsl.exe -d " & strWslDistro & " --exec dbus-launch true", 0, True
objShell.Run "wsl.exe -d " & strWslDistro & " -- bash -c ""pgrep -x sleep >/dev/null || exec sleep infinity""", 0, False

' 2. Wait up to 5 seconds for speedman to respond to socket connection
isAlive = False
For i = 1 To 10
    On Error Resume Next
    Set objHttp = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    objHttp.setTimeouts 500, 500, 1000, 1000
    objHttp.Open "GET", "http://127.0.0.1:8081/health", False
    objHttp.Send
    If Err.Number = 0 Then
        If objHttp.Status = 200 Then
            isAlive = True
            On Error GoTo 0
            Exit For
        End If
    End If
    On Error GoTo 0
    WScript.Sleep 500
Next

' 3. Open dedicated app window if Chrome or Brave exists, otherwise default browser
chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
bravePath = "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

If objFso.FileExists(chromePath) Then
    objShell.Run """" & chromePath & """ --app=" & strUrl, 1, False
ElseIf objFso.FileExists(bravePath) Then
    objShell.Run """" & bravePath & """ --app=" & strUrl, 1, False
Else
    objShell.Run strUrl, 1, False
End If

Set objHttp = Nothing
Set objFso = Nothing
Set objShell = Nothing
