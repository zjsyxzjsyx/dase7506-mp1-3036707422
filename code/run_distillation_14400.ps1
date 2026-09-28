param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][double]$SoftWeight,
    [Parameter(Mandatory=$true)][int]$Seed,
    [switch]$SkipTraining
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runDir = Join-Path 'runs/distillation-14400' $Name
$gpuPython = '.\.venv-gpu\Scripts\python.exe'
$cpuPython = '.\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path 'runs/distillation-14400' -Force | Out-Null

if (-not $SkipTraining) {
    $arguments = @('train_distilled.py', '--config', 'configs/gelu_width256.json',
        '--run-dir', $runDir, '--device', 'cuda', '--precision', 'fp32',
        '--threads', '4', '--seed', "$Seed", '--steps', '14400',
        '--batch-size', '32', '--eval-every', '300', '--peak-lr', '0.001',
        '--min-lr-ratio', '0.1', '--soft-weight', "$SoftWeight",
        '--save-checkpoints-from', '12600')
    if ($SoftWeight -gt 0) {
        $arguments += @('--teacher', 'runs/temperature-3/t105/checkpoint.pt')
    }
    $log = Join-Path 'runs/distillation-14400' "$Name-train.log"
    & $gpuPython @arguments *> $log
    if ($LASTEXITCODE -ne 0) { throw "Training failed; see $log" }
}

if (-not (Test-Path -LiteralPath (Join-Path $runDir 'metrics.json'))) {
    throw "Training metrics missing in $runDir"
}
$inputs = @()
foreach ($step in @(12600, 12900, 13200, 13500, 13800, 14100, 14400)) {
    $snapshot = Join-Path $runDir ("checkpoints/step{0:D5}.pt" -f $step)
    if (-not (Test-Path -LiteralPath $snapshot)) { throw "Missing $snapshot" }
    $inputs += @('--input', $snapshot)
}
$ema = Join-Path $runDir 'ema07.pt'
if (-not (Test-Path -LiteralPath $ema)) {
    & $cpuPython average_checkpoints.py @inputs --method ema --decay 0.7 --output $ema
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint averaging failed for $Name" }
}
foreach ($variant in @(@('final', (Join-Path $runDir 'checkpoint.pt')),
                       @('ema07', $ema))) {
    $output = Join-Path $runDir ("validation_cpu_fp32_{0}.json" -f $variant[0])
    if (-not (Test-Path -LiteralPath $output)) {
        & $cpuPython baseline_pipeline.py measure --checkpoint $variant[1] --split validation --output $output
        if ($LASTEXITCODE -ne 0) { throw "CPU validation failed for $Name $($variant[0])" }
    }
}
$final = Get-Content -LiteralPath (Join-Path $runDir 'validation_cpu_fp32_final.json') -Raw | ConvertFrom-Json
$averaged = Get-Content -LiteralPath (Join-Path $runDir 'validation_cpu_fp32_ema07.json') -Raw | ConvertFrom-Json
Write-Output "$Name final=$($final.bpb) ema07=$($averaged.bpb)"
