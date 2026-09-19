param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         Speedman 5x Fast Speech Compression" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Input: $InputPath"
Write-Host ""

try {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -Method Get -TimeoutSec 2
    } catch {
        Write-Host "Waking Speedman socket..." -ForegroundColor Yellow
        Start-Process wsl.exe -ArgumentList "-d Ubuntu-24.04 curl -s http://127.0.0.1:8081/health" -WindowStyle Hidden
        Start-Sleep -Seconds 1
    }

    Write-Host "Sending compression request to Speedman (127.0.0.1:8081)..." -ForegroundColor Cyan
    $bodyObj = @{
        input_path = $InputPath
        speed = 5.0
        preset = "fast"
        uniform = $false
    }
    $body = $bodyObj | ConvertTo-Json

    $res = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/compress/json" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600

    $silencePct = [math]::Round($res.silence_fraction * 100)

    Write-Host ""
    Write-Host "Compression Complete!" -ForegroundColor Green
    Write-Host ("  Original Duration : {0}s" -f $res.input_duration_s)
    Write-Host ("  Sped Duration     : {0}s ({1}x reduction)" -f $res.output_duration_s, $res.compression_ratio)
    Write-Host ("  Silence Fraction  : {0}%" -f $silencePct)
    Write-Host ("  Effective Rate    : {0}x" -f $res.effective_speech_rate)
    Write-Host ("  Processing Time   : {0}s" -f $res.processing_time_s)
    Write-Host ""
    Write-Host ("Output File: {0}" -f $res.windows_output_path) -ForegroundColor Yellow
} catch {
    Write-Host ("Error: " + $_.Exception.Message) -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 3
