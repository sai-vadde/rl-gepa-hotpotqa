# Runs every stage in the order used for the reported results, unattended (Windows PowerShell).
# Every stage is resumable, so running this script again continues where it stopped.
# Status goes to logs\chain_status.txt. Edit $M if your Ollama tag for the model differs.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = '.venv\Scripts\python.exe'
$M = 'qwen3:4b-instruct-2507'
$api = 'htt<path>
New-Item -ItemType Directory -Force logs | Out-Null
$stages = @(
    @{ name = 'pilot';    args = @('--stage', 'pilot', '--model', $M, '--api-base', $api) },
    @{ name = 'split';    args = @('--stage', 'split') },
    @{ name = 'baseline'; args = @('--stage', 'baseline', '--model', $M, '--api-base', $api) },
    @{ name = 'mipro';    args = @('--stage', 'mipro', '--model', $M, '--api-base', $api) },
    @{ name = 'gepa_seed11'; args = @('--stage', 'gepa', '--model', $M, '--api-base', $api, '--budget', '900', '--seed', '11') },
    @{ name = 'grpo';     args = @('--stage', 'grpo', '--model', $M, '--api-base', $api) },
    @{ name = 'tables';   args = @('--stage', 'tables') }
)
foreach ($s in $stages) {
    $t0 = Get-Date
    "$(Get-Date -Format s) START $($s.name)" | Add-Content logs\chain_status.txt
    & $py run_all.py @($s.args) *> "logs\$($s.name).log"
    $code = $LASTEXITCODE
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    "$(Get-Date -Format s) END $($s.name) exit=$code minutes=$min" | Add-Content logs\chain_status.txt
    if ($code -ne 0) { "CHAIN STOPPED at $($s.name) (exit $code)"; exit $code }
}
"CHAIN DONE"
