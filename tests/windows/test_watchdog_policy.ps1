#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PolicyPath = Join-Path $ScriptDir "..\..\scripts\service\windows\lib\watchdog-policy.ps1"

. $PolicyPath

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)] $Actual,
        [Parameter(Mandatory = $true)] $Expected,
        [Parameter(Mandatory = $true)] [string] $Message
    )

    if ($Actual -ne $Expected) {
        throw "$Message - expected '$Expected', got '$Actual'"
    }
}

$SlowStart = Get-McpWatchdogDecision `
    -Healthy $false `
    -ReadinessObserved $false `
    -UptimeSeconds 188 `
    -ConsecutiveFailures 0
Assert-Equal $SlowStart.Action "WaitForReadiness" "A legitimate 188-second cold start must not be killed"
Assert-Equal $SlowStart.ConsecutiveFailures 0 "Startup failures must not count as zombie failures"

$StartupTimeout = Get-McpWatchdogDecision `
    -Healthy $false `
    -ReadinessObserved $false `
    -UptimeSeconds 300 `
    -ConsecutiveFailures 0
Assert-Equal $StartupTimeout.Action "RestartStartupTimeout" "A never-ready process must be restarted at the startup timeout"

$Ready = Get-McpWatchdogDecision `
    -Healthy $true `
    -ReadinessObserved $false `
    -UptimeSeconds 210 `
    -ConsecutiveFailures 2
Assert-Equal $Ready.Action "Healthy" "The first successful probe must mark the process ready"
Assert-Equal $Ready.ReadinessObserved $true "Readiness must remain observable"
Assert-Equal $Ready.ConsecutiveFailures 0 "A successful probe must reset health failures"

$FirstRuntimeFailure = Get-McpWatchdogDecision `
    -Healthy $false `
    -ReadinessObserved $true `
    -UptimeSeconds 360 `
    -ConsecutiveFailures 0
Assert-Equal $FirstRuntimeFailure.Action "WaitForHealthRecovery" "A single post-readiness failure must not restart the server"
Assert-Equal $FirstRuntimeFailure.ConsecutiveFailures 1 "The first runtime failure must be counted"

$ThirdRuntimeFailure = Get-McpWatchdogDecision `
    -Healthy $false `
    -ReadinessObserved $true `
    -UptimeSeconds 420 `
    -ConsecutiveFailures 2
Assert-Equal $ThirdRuntimeFailure.Action "RestartUnhealthy" "Three post-readiness failures must restart a zombie"
Assert-Equal $ThirdRuntimeFailure.ConsecutiveFailures 3 "The restart decision must expose the failure count"

$WrapperPath = Join-Path $ScriptDir "..\..\scripts\service\windows\run_http_server_background.ps1"
$WrapperSource = Get-Content $WrapperPath -Raw
if ($WrapperSource -notmatch 'Max restart attempts[\s\S]+?exit 1') {
    throw "The wrapper must return exit code 1 after exhausting restart attempts"
}

Write-Host "watchdog-policy: PASS"
