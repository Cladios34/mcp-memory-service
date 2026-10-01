#Requires -Version 5.1

function Get-McpWatchdogDecision {
    param(
        [Parameter(Mandatory = $true)] [bool] $Healthy,
        [Parameter(Mandatory = $true)] [bool] $ReadinessObserved,
        [Parameter(Mandatory = $true)] [int] $UptimeSeconds,
        [Parameter(Mandatory = $true)] [int] $ConsecutiveFailures,
        [int] $StartupTimeoutSeconds = 300,
        [int] $MaxConsecutiveFailures = 3
    )

    if ($Healthy) {
        return [pscustomobject]@{
            Action = "Healthy"
            ReadinessObserved = $true
            ConsecutiveFailures = 0
        }
    }

    if (-not $ReadinessObserved) {
        if ($UptimeSeconds -lt $StartupTimeoutSeconds) {
            return [pscustomobject]@{
                Action = "WaitForReadiness"
                ReadinessObserved = $false
                ConsecutiveFailures = 0
            }
        }

        return [pscustomobject]@{
            Action = "RestartStartupTimeout"
            ReadinessObserved = $false
            ConsecutiveFailures = 0
        }
    }

    $UpdatedFailures = $ConsecutiveFailures + 1
    return [pscustomobject]@{
        Action = if ($UpdatedFailures -ge $MaxConsecutiveFailures) { "RestartUnhealthy" } else { "WaitForHealthRecovery" }
        ReadinessObserved = $true
        ConsecutiveFailures = $UpdatedFailures
    }
}
