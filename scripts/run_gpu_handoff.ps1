[CmdletBinding()]
param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$ArtifactRoot = "",
    [string]$Python = "python",
    [string]$AnalysisPython = "",
    [int]$Seed = 42,
    [int]$ExpectedTrainMatchedCount = 14253,
    [int]$ExpectedDevMatchedCount = 1500,
    [string]$ExpectedBaseCommit = "",
    [switch]$Resume,
    [string]$DatasetRevision = "898f51582750fe244693794f22e3f4b32c5baf95",
    [string]$WhisperRevision = "c4e2b47d88ae8b3ee0a605e09863b93aafca72e3",
    [string]$NllbRevision = "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    [string]$AfriNllbRevision = "53b1bf8d09454d092a474a8e78d5c95a32b53154",
    [string]$CometRevision = "6e64e0a56ce69524c67f304b092725687a362ef8",
    [string]$ExpectedTrainAsrSha256 = "",
    [string]$ExpectedDevAsrSha256 = "",
    [string]$ExpectedResearchTrainSha256 = "",
    [string]$ExpectedResearchValidationSha256 = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:MPLBACKEND = "Agg"

if (-not $ArtifactRoot) {
    $ArtifactRoot = Join-Path $RepositoryRoot "results\gpu-handoff-private"
}
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$ArtifactRoot = [IO.Path]::GetFullPath($ArtifactRoot)
$Generated = Join-Path $ArtifactRoot "generated"
$Outputs = Join-Path $ArtifactRoot "outputs"
$Results = Join-Path $ArtifactRoot "results"
$CometCache = Join-Path $ArtifactRoot ".hf-cache-final"
$CommandsPath = Join-Path $Results "executed_commands.jsonl"
$StatusPath = Join-Path $Results "run_status.json"

New-Item -ItemType Directory -Force -Path $Generated, $Outputs, $Results | Out-Null
Set-Location -LiteralPath $RepositoryRoot
if (-not $AnalysisPython) {
    $AnalysisPython = Join-Path $RepositoryRoot ".venv-analysis\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $AnalysisPython -PathType Leaf)) {
    throw "Isolated COMET analysis Python not found: $AnalysisPython"
}

function Write-Status {
    param([string]$Stage, [string]$State, [string]$Details = "")
    [ordered]@{
        stage = $Stage
        state = $State
        details = $Details
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments, [string]$Executable = "")
    if (-not $Executable) { $Executable = $Python }
    $record = [ordered]@{
        stage = $Name
        executable = $Executable
        arguments = $Arguments
        started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath $CommandsPath -Encoding utf8
    Write-Status -Stage $Name -State "running"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $Name failed with exit code $LASTEXITCODE."
    }
    Write-Status -Stage $Name -State "completed"
}

function Assert-Hash {
    param([string]$Path, [string]$Expected)
    if (-not $Expected) { return }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Expected hash target does not exist: $Path"
    }
    $Observed = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Expected; observed $Observed."
    }
}

function Get-DirtyPatchHash {
    $Diff = (& git -c core.safecrlf=false diff --no-ext-diff --full-index --binary HEAD) -join "`n"
    $Normalized = $Diff.Replace("`r`n", "`n").TrimEnd("`n") + "`n"
    $Bytes = [Text.Encoding]::UTF8.GetBytes($Normalized)
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}

try {
    $BaseCommit = (& git rev-parse HEAD).Trim()
    if ($ExpectedBaseCommit -and $BaseCommit -ne $ExpectedBaseCommit) {
        throw "Unexpected Git commit. Expected $ExpectedBaseCommit; observed $BaseCommit."
    }
    $DirtyStatus = (& git status --short) -join "`n"
    $Dirty = [bool]$DirtyStatus
    $Untracked = @(& git ls-files --others --exclude-standard)
    if ($Untracked.Count -gt 0) {
        throw "Untracked files make a complete dirty-patch hash ambiguous: $($Untracked -join ', ')"
    }
    $DirtyPatchSha256 = if ($Dirty) { Get-DirtyPatchHash } else { $null }
    $DirtyPatchFiles = if ($Dirty) { @(& git diff --name-only HEAD) } else { @() }

    Invoke-Step -Name "preflight_setup" -Arguments @(
        "experiments/gpu_preflight.py", "--stage", "setup", "--require-cuda",
        "--json-output", (Join-Path $Results "preflight_setup.json")
    )

    $TrainPairs = Join-Path $Generated "naijas2st_train_pairs_all.jsonl"
    $TrainAsr = Join-Path $Generated "naijas2st_train_whisper_all.jsonl"
    $Train = Join-Path $Generated "research_train.jsonl"
    $Validation = Join-Path $Generated "research_val.jsonl"
    $SplitStats = Join-Path $Results "split_stats.json"

    Invoke-Step -Name "prepare_train_pairs" -Arguments @(
        "experiments/prepare_naijas2st_pairs.py", "--split", "train",
        "--max-samples", "0", "--dataset-revision", $DatasetRevision,
        "--output", $TrainPairs
    )
    $AsrArgs = @(
        "experiments/generate_asr_noise.py", "--pairs", $TrainPairs,
        "--split", "train", "--max-samples", "0",
        "--target-matched-count", "$ExpectedTrainMatchedCount", "--shuffle-buffer", "0",
        "--seed", "$Seed", "--dataset-revision", $DatasetRevision,
        "--asr-model-revision", $WhisperRevision, "--output", $TrainAsr
    )
    if ($Resume) { $AsrArgs += "--resume" }
    Invoke-Step -Name "generate_train_asr" -Arguments $AsrArgs
    Assert-Hash -Path $TrainAsr -Expected $ExpectedTrainAsrSha256

    Invoke-Step -Name "make_research_splits" -Arguments @(
        "experiments/make_research_splits.py", "--input", $TrainAsr,
        "--train-output", $Train, "--val-output", $Validation,
        "--val-fraction", "0.10", "--seed", "$Seed",
        "--stats-output", $SplitStats
    )
    Assert-Hash -Path $Train -Expected $ExpectedResearchTrainSha256
    Assert-Hash -Path $Validation -Expected $ExpectedResearchValidationSha256
    $Split = Get-Content -LiteralPath $SplitStats -Raw | ConvertFrom-Json
    if ($Split.alignment_overlap -ne 0 -or $Split.exact_bilingual_text_overlap -ne 0) {
        throw "Leakage gate failed."
    }
    if (($Split.train.rows + $Split.validation.rows) -ne $Split.input_rows) {
        throw "Split row conservation failed."
    }

    Invoke-Step -Name "preflight_train" -Arguments @(
        "experiments/gpu_preflight.py", "--stage", "train", "--require-cuda",
        "--required-path", $Train, "--required-path", $Validation,
        "--json-output", (Join-Path $Results "preflight_train.json")
    )
    foreach ($Mode in @("clean", "noisy", "mixed")) {
        Invoke-Step -Name "train_$Mode" -Arguments @(
            "experiments/train_error_aware_controlled.py", "--train-jsonl", $Train,
            "--val-jsonl", $Validation, "--mode", $Mode,
            "--output-dir", (Join-Path $Outputs $Mode), "--epochs", "3",
            "--batch-size", "2", "--gradient-accumulation", "4",
            "--learning-rate", "2e-4", "--seed", "$Seed",
            "--base-model-revision", $NllbRevision
        )
    }

    $Metadata = @{}
    foreach ($Mode in @("clean", "noisy", "mixed")) {
        $Metadata[$Mode] = Get-Content -LiteralPath (Join-Path $Outputs "$Mode\experiment_metadata.json") -Raw | ConvertFrom-Json
    }
    if ($Metadata.clean.train_examples -ne $Metadata.noisy.train_examples -or
        $Metadata.clean.train_examples -ne $Metadata.mixed.train_examples) {
        throw "Training exposure counts differ across conditions."
    }
    $ExpectedExposure = $Metadata.clean.train_examples
    if ($Metadata.clean.train_source_counts.clean -ne $ExpectedExposure -or
        $Metadata.noisy.train_source_counts.asr_noise -ne $ExpectedExposure -or
        ($Metadata.mixed.train_source_counts.clean + $Metadata.mixed.train_source_counts.asr_noise) -ne $ExpectedExposure -or
        $Metadata.mixed.train_source_counts.clean -ne $Metadata.mixed.train_source_counts.asr_noise) {
        throw "Clean/noisy/mixed source exposure is inconsistent."
    }

    $DevPairs = Join-Path $Generated "naijas2st_dev_pairs_all.jsonl"
    $DevAsr = Join-Path $Generated "naijas2st_dev_whisper_all.jsonl"
    Invoke-Step -Name "prepare_dev_pairs" -Arguments @(
        "experiments/prepare_naijas2st_pairs.py", "--split", "dev",
        "--max-samples", "0", "--dataset-revision", $DatasetRevision,
        "--output", $DevPairs
    )
    $DevArgs = @(
        "experiments/generate_asr_noise.py", "--pairs", $DevPairs,
        "--split", "dev", "--target-matched-count", "$ExpectedDevMatchedCount",
        "--max-samples", "0", "--shuffle-buffer", "0", "--seed", "$Seed",
        "--dataset-revision", $DatasetRevision,
        "--asr-model-revision", $WhisperRevision, "--output", $DevAsr
    )
    if ($Resume) { $DevArgs += "--resume" }
    Invoke-Step -Name "generate_dev_asr" -Arguments $DevArgs
    Assert-Hash -Path $DevAsr -Expected $ExpectedDevAsrSha256

    $Suite = Join-Path $Results "whisper_error_aware_suite.csv"
    Invoke-Step -Name "evaluate_whisper_suite" -Arguments @(
        "experiments/evaluate_research_suite.py", "--input", $DevAsr,
        "--source-field", "hausa_asr", "--batch-size", "4",
        "--baseline-revision", "nllb=$NllbRevision",
        "--baseline-revision", "afrinllb=$AfriNllbRevision",
        "--adapter", "clean=$(Join-Path $Outputs 'clean')",
        "--adapter", "noisy=$(Join-Path $Outputs 'noisy')",
        "--adapter", "mixed=$(Join-Path $Outputs 'mixed')",
        "--adapter-base-revision", $NllbRevision, "--output", $Suite
    )
    Invoke-Step -Name "analysis_import_smoke" -Executable $AnalysisPython -Arguments @(
        "-c", "from importlib.metadata import version; import comet, torchmetrics; assert version('unbabel-comet') == '2.2.7'"
    )
    Invoke-Step -Name "analyze_predictions" -Executable $AnalysisPython -Arguments @(
        "analysis/analyze_predictions.py", "--input", $Suite,
        "--output-dir", (Join-Path $Results "analysis_whisper"),
        "--baseline", "nllb", "--candidate", "mixed", "--bootstrap", "1000",
        "--seed", "$Seed", "--qualitative-n", "40", "--comet",
        "--comet-revision", $CometRevision, "--comet-cache-dir", $CometCache,
        "--comet-batch-size", "1"
    )

    $Environment = & $Python -c "import json,platform,torch,transformers,datasets,peft,sacrebleu; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'transformers':transformers.__version__,'datasets':datasets.__version__,'peft':peft.__version__,'sacrebleu':sacrebleu.__version__}))"
    $Gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    [ordered]@{
        base_commit = $BaseCommit
        dirty_worktree = $Dirty
        dirty_patch_sha256 = $DirtyPatchSha256
        dirty_patch_files = $DirtyPatchFiles
        seed = $Seed
        revisions = [ordered]@{
            dataset = $DatasetRevision; whisper = $WhisperRevision
            nllb = $NllbRevision; afrinllb = $AfriNllbRevision; comet = $CometRevision
        }
        environment = $Environment | ConvertFrom-Json
        analysis_python = (& $AnalysisPython -c "import platform; print(platform.python_version())").Trim()
        gpu = $Gpu
        completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Results "run_manifest.json") -Encoding utf8

    Write-Status -Stage "complete" -State "completed"
}
catch {
    Write-Status -Stage "failed" -State "failed" -Details $_.Exception.Message
    throw
}
