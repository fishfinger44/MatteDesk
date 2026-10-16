# tools/server_port.ps1 - used by start_v….bat: is the MatteDesk port taken, and by what?
#   -Check : exit 0 = port free, 1 = an older MatteDesk server (uvicorn app:app) holds it, 2 = another program holds it
#   -Stop  : stop the MatteDesk server(s) on this port (uvicorn app:app ... --port <Port>, also its .venv launcher)
param([int]$Port = 8765, [switch]$Check, [switch]$Stop)

$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $conn) { exit 0 }
$owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn[0].OwningProcess)"
$isMd = $owner -and $owner.CommandLine -like '*uvicorn app:app*'

if ($Check) {
    if ($isMd) { exit 1 }
    $name = if ($owner) { $owner.Name } else { "PID $($conn[0].OwningProcess)" }
    Write-Host "[MatteDesk] Port $Port is used by another program: $name"
    exit 2
}

if ($Stop) {
    if (-not $isMd) { exit 2 }
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn app:app*' -and $_.CommandLine -like "*--port $Port*" }
    foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt 20; $i++) {
        if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { exit 0 }
        Start-Sleep -Milliseconds 250
    }
    exit 1
}
exit 0
