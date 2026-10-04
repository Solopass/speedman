param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,
    [switch]$Transcribe = $false,
    [switch]$Obsidian = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         Speedman 5x Fast Speech Compression" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Input: $InputPath"
if ($Transcribe) {
    Write-Host "Auto-Transcribe (Parakeet ASR): Enabled" -ForegroundColor Magenta
}
if ($Obsidian) {
    Write-Host "Auto-Obsidian Note: Enabled" -ForegroundColor Magenta
}
Write-Host ""

try {
    $ready = $false
    for ($i = 0; $i -lt 8; $i++) {
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -Method Get -TimeoutSec 3
            if ($h.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {
            if ($i -eq 0) {
                Write-Host "Waking Speedman socket..." -ForegroundColor Yellow
                Start-Process wsl.exe -ArgumentList "-d Ubuntu-24.04 --exec dbus-launch true" -WindowStyle Hidden -Wait
            }
            Start-Sleep -Milliseconds 1000
        }
    }

    Write-Host "Sending compression request to Speedman (127.0.0.1:8081)..." -ForegroundColor Cyan
    $bodyObj = @{
        input_path = $InputPath
        speed = 5.0
        preset = "fast"
        uniform = $false
        auto_obsidian = [bool]($Obsidian -and -not $Transcribe)
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

    if ($Transcribe) {
        Write-Host ""
        Write-Host "--------------------------------------------------------" -ForegroundColor Cyan
        Write-Host "[1/3] Checking Media API (127.0.0.1:8080)..." -ForegroundColor Cyan

        $mediaOnline = $false
        try {
            $stat = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/media/status" -Method Get -TimeoutSec 3
            $mediaOnline = [bool]$stat.online
        } catch {
            $mediaOnline = $false
        }

        if (-not $mediaOnline) {
            Write-Host "  Media API is offline. Skipping transcription." -ForegroundColor Yellow
            Write-Host "  (Ensure Media API is running on port 8080 to enable Parakeet ASR)" -ForegroundColor DarkGray
        } else {
            Write-Host "  Media API is online. Initiating Parakeet ASR on 1x source audio..." -ForegroundColor Green
            $transcribeBody = @{
                engine = "parakeet"
                sync_to_output = $res.filename
                source_path = $res.source_path
            } | ConvertTo-Json

            $escapedFilename = [System.Uri]::EscapeDataString($res.filename)
            $tRes = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/transcribe/$escapedFilename" -Method Post -Body $transcribeBody -ContentType "application/json" -TimeoutSec 15
            $jobId = $tRes.job_id

            if ($jobId) {
                Write-Host "  Job dispatched: $jobId" -ForegroundColor DarkCyan
                Write-Host "  Transcribing with Parakeet" -NoNewline -ForegroundColor Cyan

                $pollInterval = 2
                $maxWaitSec = 600
                $elapsed = 0
                $jobCompleted = $false

                while ($elapsed -lt $maxWaitSec) {
                    Start-Sleep -Seconds $pollInterval
                    $elapsed += $pollInterval
                    Write-Host "." -NoNewline -ForegroundColor Cyan

                    try {
                        $statusRes = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/transcribe/status/$jobId" -Method Get -TimeoutSec 5
                        if ($statusRes.status -eq "completed") {
                            $jobCompleted = $true
                            break
                        } elseif ($statusRes.status -eq "failed" -or $statusRes.status -eq "error") {
                            Write-Host " Failed!" -ForegroundColor Red
                            Write-Host ("  Transcription failed: " + $statusRes.error) -ForegroundColor Red
                            break
                        }
                    } catch {
                        # Transient polling glitch, retry next tick
                    }
                }

                if ($jobCompleted) {
                    Write-Host (" Done in {0}s!" -f $elapsed) -ForegroundColor Green

                    Write-Host "[2/3] Warping transcript onto compressed timeline..." -ForegroundColor Cyan
                    $syncBody = @{
                        output_name = $res.filename
                        job_id = $jobId
                    } | ConvertTo-Json

                    $syncRes = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/transcribe/sync" -Method Post -Body $syncBody -ContentType "application/json" -TimeoutSec 30
                    Write-Host ("  Synced Segments  : {0} cues" -f $syncRes.segments) -ForegroundColor Green
                    Write-Host ("  WebVTT Subtitles : {0}" -f $syncRes.windows_vtt_path) -ForegroundColor Yellow
                    Write-Host ("  Plain Text       : {0}" -f $syncRes.windows_txt_path) -ForegroundColor Yellow

                    Write-Host "[3/3] Detecting semantic chapters & TOC..." -ForegroundColor Cyan
                    try {
                        $chapRes = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/chapters/$escapedFilename" -Method Get -TimeoutSec 45
                        $chapCount = if ($chapRes.chapters) { $chapRes.chapters.Count } else { 0 }
                        Write-Host ("  Semantic Chapters : {0} chapters detected" -f $chapCount) -ForegroundColor Green
                    } catch {
                        Write-Host ("  (Chapter detection notice: " + $_.Exception.Message + ")") -ForegroundColor DarkGray
                    }

                    if ($Obsidian) {
                        Write-Host ""
                        Write-Host "Generating Obsidian executive summary note..." -ForegroundColor Cyan
                        $obsBody = @{
                            output_name = $res.filename
                            folder = "Summaries"
                            include_summary = $true
                            include_transcript = $true
                            include_highlights = $true
                            use_llm = $true
                        } | ConvertTo-Json

                        try {
                            $obsRes = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/export/obsidian" -Method Post -Body $obsBody -ContentType "application/json" -TimeoutSec 60
                            Write-Host ("  Obsidian Note     : {0}" -f $obsRes.windows_path) -ForegroundColor Green

                            try {
                                $openBody = @{ path = $obsRes.path } | ConvertTo-Json
                                $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/export/obsidian/open" -Method Post -Body $openBody -ContentType "application/json" -TimeoutSec 5
                                Write-Host "  Note opened in Obsidian." -ForegroundColor Green
                            } catch {
                                Write-Host ("  (Could not auto-open Obsidian: " + $_.Exception.Message + ")") -ForegroundColor DarkGray
                            }
                        } catch {
                            Write-Host ("  Obsidian export failed: " + $_.Exception.Message) -ForegroundColor Red
                        }
                    }
                }
            }
        }
    } elseif ($Obsidian -and $res.obsidian_note) {
        Write-Host ("  Obsidian Note     : {0}" -f $res.obsidian_note.windows_path) -ForegroundColor Green
        try {
            $openBody = @{ path = $res.obsidian_note.path } | ConvertTo-Json
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/api/v1/export/obsidian/open" -Method Post -Body $openBody -ContentType "application/json" -TimeoutSec 5
        } catch {
            Write-Host ("  (Could not auto-open Obsidian: " + $_.Exception.Message + ")") -ForegroundColor DarkGray
        }
    }
} catch {
    Write-Host ("Error: " + $_.Exception.Message) -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 3
