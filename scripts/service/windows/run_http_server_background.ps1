#Requires -Version 5.1
<#
.SYNOPSIS
    Wrapper script to run MCP Memory HTTP Server in background with logging and restart logic.

.DESCRIPTION
    This script is designed to be executed by Windows Task Scheduler.
    It runs the HTTP server with proper environment setup, logging, and automatic restart on failure.

.NOTES
    File Name      : run_http_server_background.ps1
    Prerequisite   : Python, uv package manager
    Location       : scripts/service/windows/
#>

# Configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item "$ScriptDir\..\..\..").FullName

# Load shared server-config helper (reads host/port/https from .env)
. "$ScriptDir\lib\server-config.ps1"
. "$ScriptDir\lib\watchdog-policy.ps1"
$ServerConfig = Get-McpServerConfig -ProjectRoot $ProjectRoot
Enable-McpSelfSignedCertBypass
$McpWebExtras = Get-McpWebRequestExtraParams -HttpsEnabled $ServerConfig.HttpsEnabled

$LogDir = Join-Path $env:LOCALAPPDATA "mcp-memory\logs"
$LogFile = Join-Path $LogDir "http-server.log"
# Separate file for the Python subprocess output (stdout+stderr). The wrapper
# log ($LogFile) holds wrapper meta info only; the Python process logs land in
# $PythonLogFile so you can tail them live and they don't get lost through
# broken .NET event handlers (the previous implementation silently dropped
# all [SERVER] lines because `$script:LogFile` isn't captured in the event
# handler runspace).
$PythonLogFile = Join-Path $LogDir "http-server-python.log"
$PidFile = Join-Path $env:LOCALAPPDATA "mcp-memory\http-server.pid"
$MaxRestarts = 3
$RestartDelaySeconds = 60
$StartupTimeoutSeconds = 300
$HealthCheckIntervalSeconds = 30
$MaxConsecutiveHealthFailures = 3

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Single-instance guard.
# GOTCHA(2026-06-09): the 5-min scheduler tick and the Claude session hooks can
# both spawn this wrapper. Overlapping instances collided on the shared log
# files ("file in use by another process"), killed each other's restart loops
# and ended in "Max restart attempts reached. Giving up." while no server ran.
# A named mutex guarantees only one wrapper manages the server lifecycle; the
# OS releases it automatically when the process exits.
$script:WrapperMutex = New-Object System.Threading.Mutex($false, "Global\MCPMemoryHTTPServerWrapper")
$MutexAcquired = $false
try {
    $MutexAcquired = $script:WrapperMutex.WaitOne(2000)
} catch [System.Threading.AbandonedMutexException] {
    # Previous holder died without releasing - we now own it.
    $MutexAcquired = $true
}
if (-not $MutexAcquired) {
    # Another wrapper instance is already managing the server. Nothing to do.
    exit 0
}

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $LogMessage

    # Keep log file under 10MB
    if ((Get-Item $LogFile -ErrorAction SilentlyContinue).Length -gt 10MB) {
        $OldLog = "$LogFile.old"
        if (Test-Path $OldLog) { Remove-Item $OldLog -Force }
        Rename-Item $LogFile $OldLog
    }
}

# Load .env file
function Load-EnvFile {
    $EnvFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $EnvFile) {
        Write-Log "Loading environment from $EnvFile"
        Get-Content $EnvFile | ForEach-Object {
            if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
                $Name = $matches[1].Trim()
                $Value = $matches[2].Trim().Trim('"').Trim("'")
                [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
            }
        }
    } else {
        Write-Log "No .env file found at $EnvFile" "WARN"
    }
}

# Find uv executable (Task Scheduler has a minimal PATH)
function Find-Executable {
    # Common install locations for uv on Windows
    $Candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:ProgramData "uv\uv.exe")
    )

    foreach ($Path in $Candidates) {
        if (Test-Path $Path) {
            Write-Log "Found uv at: $Path"
            return $Path
        }
    }

    # Try PATH resolution (works in interactive shells, may fail in Task Scheduler)
    $PathResolved = Get-Command uv -ErrorAction SilentlyContinue
    if ($PathResolved) {
        Write-Log "Found uv on PATH: $($PathResolved.Source)"
        return $PathResolved.Source
    }

    Write-Log "uv not found, will fall back to python directly" "WARN"
    return $null
}

# Find every live process belonging to OUR server (uv parent or python child),
# identified by command line - never by PID file alone.
# GOTCHA(2026-07-13): the actual port-8000 listener runs as
# "python -m uvicorn mcp_memory_service.web.app:app", NOT run_http_server.py.
# Matching only the launcher script left orphaned uvicorn children invisible:
# they kept the port and the log file, every restart attempt died on
# WinError 10048 / "file in use", and the wrapper gave up while a half-dead
# server still held the socket (incident 2026-07-13 09:31 + 13:58).
function Get-ServerProcesses {
    try {
        @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -and $_.CommandLine -match 'run_http_server\.py|uvicorn\s+mcp_memory_service\.web\.app' })
    } catch {
        @()
    }
}

# Best-effort log consolidation. Descendant Python processes can briefly keep
# redirected handles open after uv.exe exits; diagnostics must never consume a
# restart attempt merely because the log is still locked.
function Merge-PythonErrorLog {
    param(
        [Parameter(Mandatory = $true)] [string] $StandardOutputPath,
        [Parameter(Mandatory = $true)] [string] $StandardErrorPath
    )

    if (-not (Test-Path $StandardErrorPath)) { return }

    for ($Attempt = 1; $Attempt -le 10; $Attempt++) {
        try {
            if ((Get-Item $StandardErrorPath).Length -eq 0) {
                Remove-Item $StandardErrorPath -Force -ErrorAction SilentlyContinue
                return
            }

            Add-Content -Path $StandardOutputPath -Value ""
            Add-Content -Path $StandardOutputPath -Value "===== STDERR ====="
            Get-Content $StandardErrorPath | Add-Content -Path $StandardOutputPath
            Remove-Item $StandardErrorPath -Force -ErrorAction SilentlyContinue
            return
        } catch {
            if ($Attempt -lt 10) {
                Start-Sleep -Milliseconds 500
            }
        }
    }

    Write-Log "Could not merge stderr because a process still holds a log handle; preserved at $StandardErrorPath" "WARN"
}

# Check if server is already running.
# DECISION(2026-06-09): HTTP health is the ONLY proof of life. The previous
# implementation trusted the PID file first ($Process.ProcessName -like
# "*python*"): when the server died and Windows recycled that PID for any other
# python process (NotebookLM CLI spawns dozens), every 5-min tick said
# "already running" forever and the dead server was never restarted - the user
# had to reconnect the memory MCP manually at every Claude session.
function Test-ServerRunning {
    # 1. HTTP health endpoint (URL derived from .env) = source of truth
    try {
        $healthParams = @{
            Uri = $ServerConfig.HealthUrl
            TimeoutSec = 5
            UseBasicParsing = $true
            ErrorAction = 'SilentlyContinue'
        } + $McpWebExtras
        $Response = Invoke-WebRequest @healthParams
        if ($Response.StatusCode -eq 200) {
            return $true
        }
    } catch {
        # Server not responding over HTTP
    }

    # 2. HTTP is down. Is a genuine server process around?
    $ServerProcs = Get-ServerProcesses
    if ($ServerProcs.Count -gt 0) {
        $Youngest = $ServerProcs | Sort-Object CreationDate -Descending | Select-Object -First 1
        $AgeSeconds = [int]((Get-Date) - $Youngest.CreationDate).TotalSeconds

        # Hybrid cold starts can include ONNX, remote storage and scheduler
        # initialization. Do not classify a never-ready process as a zombie.
        if ($AgeSeconds -lt $StartupTimeoutSeconds) {
            Write-Log "Server process PID $($Youngest.ProcessId) started ${AgeSeconds}s ago - cold start in progress, not restarting."
            return $true
        }

        # A process that never became reachable within the bounded startup
        # window is safe to recycle. Runtime zombies are detected faster by the
        # readiness-aware loop below.
        # Kill the whole tree so the restart below binds the port cleanly.
        foreach ($Proc in $ServerProcs) {
            Write-Log "Killing unready server process tree PID $($Proc.ProcessId) (alive ${AgeSeconds}s, HTTP dead)" "WARN"
            taskkill /PID $Proc.ProcessId /T /F 2>$null | Out-Null
        }
        Start-Sleep -Seconds 2
    }

    # Stale PID file is meaningless at this point
    if (Test-Path $PidFile) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    return $false
}

# Main execution
Write-Log "========== MCP Memory HTTP Server Starting =========="
Write-Log "Project Root: $ProjectRoot"
Write-Log "Log File: $LogFile"

# Check if already running
if (Test-ServerRunning) {
    Write-Log "Server is already running. Exiting." "WARN"
    exit 0
}

# Change to project directory
Set-Location $ProjectRoot
Write-Log "Working directory: $(Get-Location)"

# Load environment variables
Load-EnvFile

# Restart loop
$RestartCount = 0
while ($RestartCount -lt $MaxRestarts) {
    # GOTCHA(2026-07-13): re-check before every retry. After a crash, an
    # orphaned uvicorn child (or a server started by another launcher, e.g. a
    # Claude session hook) may already hold port 8000. Starting blindly ended
    # in 10048 bind failures until "Giving up" while a server was in fact
    # coming back up. Test-ServerRunning also sweeps genuine zombies.
    if ($RestartCount -gt 0 -and (Test-ServerRunning)) {
        Write-Log "A healthy server is already running (started outside this attempt). Exiting." "WARN"
        exit 0
    }
    Write-Log "Starting HTTP server (attempt $($RestartCount + 1)/$MaxRestarts)..."

    try {
        # Resolve executable (uv or python fallback)
        $UvPath = Find-Executable
        $ExePath = $null
        $ExeArgs = $null

        if ($UvPath) {
            $ExePath = $UvPath
            $ExeArgs = @("run", "python", "scripts/server/run_http_server.py")
        } else {
            # Fallback: run python directly (same PATH issue applies)
            $PythonPath = $null

            # Try py.exe launcher first (installed in SystemRoot, reliably on PATH)
            $PyLauncher = Join-Path $env:SystemRoot "py.exe"
            if (Test-Path $PyLauncher) {
                $PythonPath = $PyLauncher
            }

            # Try common Python install locations
            if (-not $PythonPath) {
                $PythonCandidates = @(
                    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python3*\python.exe"),
                    (Join-Path $env:ProgramFiles "Python3*\python.exe"),
                    "C:\Python3*\python.exe"
                )
                foreach ($Pattern in $PythonCandidates) {
                    $PyMatches = Get-Item $Pattern -ErrorAction SilentlyContinue
                    if ($PyMatches) { $PythonPath = ($PyMatches | Sort-Object DirectoryName -Descending | Select-Object -First 1).FullName; break }
                }
            }

            # Last resort: PATH resolution
            if (-not $PythonPath) {
                $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
            }

            if (-not $PythonPath) {
                Write-Log "Neither uv nor python found. Cannot start server." "ERROR"
                throw "No Python executable found"
            }

            $ExePath = $PythonPath
            $ExeArgs = @("scripts/server/run_http_server.py")
        }

        Write-Log "Executable: $ExePath $($ExeArgs -join ' ')"
        Write-Log "Python output will be written to: $PythonLogFile"

        # Rotate Python log unconditionally before each start so that the
        # previous attempt's output is preserved when the server crashes and
        # restarts. Start-Process overwrites the target file, so without this
        # the crash log from iteration N would be silently deleted by iteration N+1.
        # GOTCHA(2026-07-13): rotation must never kill the whole start attempt.
        # An orphaned child can still hold the log handle right after a kill;
        # the resulting "file in use" exception used to land in the outer catch
        # and burn one of the 3 restart attempts without even trying to start.
        try {
            if (Test-Path $PythonLogFile) {
                $OldPythonLog = "$PythonLogFile.old"
                if (Test-Path $OldPythonLog) { Remove-Item $OldPythonLog -Force }
                Rename-Item $PythonLogFile $OldPythonLog
            }
        } catch {
            $PythonLogFile = Join-Path $LogDir ("http-server-python.{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
            Write-Log "Python log locked by another process; using $PythonLogFile for this attempt" "WARN"
        }

        # Start-Process cannot merge streams. Use an immutable per-attempt
        # stderr path so a slow-closing child can never block the next launch.
        $PythonErrFile = Join-Path $LogDir ("http-server-python.stderr.{0}.attempt{1}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff"), ($RestartCount + 1))

        # Start-Process is the PowerShell-idiomatic way to redirect output to a
        # file. It avoids the .NET event handler runspace bug where
        # `$script:LogFile` was not captured, which silently dropped every
        # line of server output in the previous implementation.
        $Process = Start-Process -FilePath $ExePath `
            -ArgumentList $ExeArgs `
            -WorkingDirectory $ProjectRoot `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput $PythonLogFile `
            -RedirectStandardError $PythonErrFile

        # Save PID
        Set-Content -Path $PidFile -Value $Process.Id
        Write-Log "Server started with PID $($Process.Id)"

        # Watchdog loop: wait for exit OR detect a zombie (process alive, HTTP
        # dead). GOTCHA(2026-06-09): after sleep/resume the uvicorn listener can
        # die while the python process survives; the old blocking WaitForExit
        # never noticed, the wrapper held the single-instance mutex forever and
        # the dead server was never restarted.
        $ConsecutiveHealthFailures = 0
        $ReadinessObserved = $false
        while (-not $Process.HasExited) {
            if ($Process.WaitForExit($HealthCheckIntervalSeconds * 1000)) { break }

            $UptimeSeconds = [int]((Get-Date) - $Process.StartTime).TotalSeconds
            $Healthy = $false
            try {
                $healthParams = @{
                    Uri = $ServerConfig.HealthUrl
                    TimeoutSec = 5
                    UseBasicParsing = $true
                } + $McpWebExtras
                $Response = Invoke-WebRequest @healthParams
                if ($Response.StatusCode -eq 200) { $Healthy = $true }
            } catch {
                # HTTP dead while process alive
            }

            $Decision = Get-McpWatchdogDecision `
                -Healthy $Healthy `
                -ReadinessObserved $ReadinessObserved `
                -UptimeSeconds $UptimeSeconds `
                -ConsecutiveFailures $ConsecutiveHealthFailures `
                -StartupTimeoutSeconds $StartupTimeoutSeconds `
                -MaxConsecutiveFailures $MaxConsecutiveHealthFailures

            $WasReady = $ReadinessObserved
            $ReadinessObserved = $Decision.ReadinessObserved
            $ConsecutiveHealthFailures = $Decision.ConsecutiveFailures

            switch ($Decision.Action) {
                "Healthy" {
                    if (-not $WasReady) {
                        Write-Log "Server readiness confirmed after ${UptimeSeconds}s."
                    }
                }
                "WaitForReadiness" {
                    Write-Log "Waiting for server readiness (${UptimeSeconds}s/$($StartupTimeoutSeconds)s)."
                }
                "WaitForHealthRecovery" {
                    Write-Log "Health check failed ($ConsecutiveHealthFailures/$MaxConsecutiveHealthFailures) after readiness while server PID $($Process.Id) is alive" "WARN"
                }
                "RestartStartupTimeout" {
                    Write-Log "Server did not become ready within $StartupTimeoutSeconds seconds. Killing process tree to restart." "ERROR"
                    taskkill /PID $Process.Id /T /F 2>$null | Out-Null
                }
                "RestartUnhealthy" {
                    Write-Log "Server became unhealthy after readiness ($ConsecutiveHealthFailures consecutive failures). Killing process tree to restart." "ERROR"
                    taskkill /PID $Process.Id /T /F 2>$null | Out-Null
                }
            }
        }

        # Final blocking wait so the exit code is reliably available
        $Process.WaitForExit()
        $ExitCode = $Process.ExitCode

        Merge-PythonErrorLog -StandardOutputPath $PythonLogFile -StandardErrorPath $PythonErrFile

        Write-Log "Server exited with code $ExitCode" $(if ($ExitCode -eq 0) { "INFO" } else { "ERROR" })

        # Clean up PID file
        if (Test-Path $PidFile) {
            Remove-Item $PidFile -Force
        }

        # If clean exit (0), don't restart
        if ($ExitCode -eq 0) {
            Write-Log "Server stopped gracefully. Not restarting."
            break
        }

    } catch {
        Write-Log "Error starting server: $_" "ERROR"
    }

    $RestartCount++

    if ($RestartCount -lt $MaxRestarts) {
        Write-Log "Waiting $RestartDelaySeconds seconds before restart..."
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}

if ($RestartCount -ge $MaxRestarts) {
    Write-Log "Max restart attempts ($MaxRestarts) reached. Giving up." "ERROR"
    exit 1
}

Write-Log "========== MCP Memory HTTP Server Stopped =========="
