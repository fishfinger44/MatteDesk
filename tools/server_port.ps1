# tools/server_port.ps1 - used by start_v....bat and tools/launch_app.py: is the MatteDesk port taken, and by what?
#   -Check : exit 0 = port free, 1 = an older MatteDesk server (uvicorn app:app) holds it, 2 = another program holds it
#   -Stop  : stop the MatteDesk server(s) on this port (uvicorn app:app ... --port <Port>, also its .venv launcher)
# The TCP table is not enough on Windows (seen 2026-11-10): the listening socket of the .venv launcher -> base Python
# pair was reported with owner PID 0 ("System Idle Process") or not listed at all while the server still answered, so
# the old server was never stopped. A MatteDesk server is therefore also found by its command line and by
# GET /api/version on the port.
param([int]$Port = 8765, [switch]$Check, [switch]$Stop)

function MdProcs { @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn app:app*' -and $_.CommandLine -like "*--port $Port*" }) }
function Answers {
    try {
        $v = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/version" -TimeoutSec 2 -ErrorAction Stop
        return [bool]($v.version)
    } catch { return $false }
}
function Listening { @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) }

$conn = Listening
$md = MdProcs
$alive = Answers
if (-not $conn -and -not $md -and -not $alive) { exit 0 }

$owners = @($conn | ForEach-Object { $_.OwningProcess } | Where-Object { $_ -gt 4 } | Select-Object -Unique)
$owner = if ($owners) { Get-CimInstance Win32_Process -Filter "ProcessId=$($owners[0])" } else { $null }
if ($owner) { $isMd = $owner.CommandLine -like '*uvicorn app:app*' }
else { $isMd = ($md.Count -gt 0) -or $alive }      # orphaned / hidden listener: MatteDesk when its server is there

if ($Check) {
    if ($isMd) { exit 1 }
    $name = if ($owner) { $owner.Name } else { "PID $($conn[0].OwningProcess)" }
    Write-Host "[MatteDesk] Port $Port is used by another program: $name"
    exit 2
}

if ($Stop) {
    if (-not $isMd) { exit 2 }
    foreach ($p in $md) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt 40; $i++) {
        if (-not (Listening) -and -not (MdProcs) -and -not (Answers)) { exit 0 }
        Start-Sleep -Milliseconds 250
    }
    exit 1
}
exit 0
