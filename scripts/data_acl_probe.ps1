# SafeDrive v3.4.4 access probe in plain PowerShell, for accounts that cannot run the project's Python (its interpreter lives in
# the owner's profile). Same output as `python scripts/data_acl.py probe`: acl_probes\probe_<user>_<time>.json (outside data\).
# Run it AS THE ACCOUNT BEING TESTED. It only tries to list, read one byte and create + delete a test file; it changes nothing.
param([string]$Data = "D:\SanphambpAI\data", [string]$Out = "D:\SanphambpAI\acl_probes")
$sess = Join-Path $Data "sessions"
$user = (whoami).Trim()
$groups = whoami /groups
$rec = [ordered]@{
    at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"); user = $user; tool = "data_acl_probe.ps1"
    elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    admin_member = [bool]($groups -match "S-1-5-32-544"); target = $sess
}
try { $names = Get-ChildItem -LiteralPath $sess -ErrorAction Stop | ForEach-Object Name; $rec.can_list_sessions = $true }
catch { $names = @(); $rec.can_list_sessions = $false; $rec.list_error = $_.Exception.GetType().Name }
$raw = $null
foreach ($n in $names) { $v = Get-ChildItem -LiteralPath (Join-Path $sess "$n\video") -Filter "segment_*.mp4" -ErrorAction SilentlyContinue | Select-Object -First 1; if ($v) { $raw = $v.FullName; break } }
if (-not $raw) {
    # without listing rights, try a raw video path known from the owner's side (passed by the test harness)
    $raw = $env:SAFEDRIVE_PROBE_FILE
}
if ($raw) {
    try { $fs = [IO.File]::OpenRead($raw); [void]$fs.ReadByte(); $fs.Close(); $rec.can_read_raw = $true }
    catch { $rec.can_read_raw = $false; $rec.read_error = $_.Exception.GetType().Name }
    $rec.raw_file_tested = $raw.Replace($Data + "\", "")
} else { $rec.can_read_raw = $false }
$t = Join-Path $sess (".acl_probe_" + $PID + ".tmp")
try { [IO.File]::WriteAllText($t, "x"); Remove-Item -LiteralPath $t -Force; $rec.can_write = $true }
catch { $rec.can_write = $false; $rec.write_error = $_.Exception.GetType().Name }
$rec.outcome = if (-not $rec.can_list_sessions -and -not $rec.can_read_raw -and -not $rec.can_write) { "denied" } else { "allowed" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$safe = ($user -replace "[^A-Za-z0-9]", "_")
$rec | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $Out ("probe_" + $safe + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json"))
$rec | ConvertTo-Json
