# End-to-end pipeline (Windows PowerShell). Each stage's console output is saved in results/logs/.
# Usage:  .\run_all.ps1            (assumes .venv exists, see README)
# "Continue": Python writes warnings to stderr, which PowerShell 5.1 would otherwise treat as
# terminating errors. Failures are detected through the process exit code instead.
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force results\logs | Out-Null

function Invoke-Stage($name, [string[]]$cmd) {
    Write-Host "`n==== $name ====" -ForegroundColor Cyan
    & $py -u @cmd 2>&1 | Tee-Object -FilePath "results\logs\$name.log"
    if ($LASTEXITCODE -ne 0) { throw "stage $name failed" }
}

# ---------------- Question 1: knowledge distillation (CIFAR-100) ----------------
Invoke-Stage "01_teacher"   @("-m", "q1_kd.train_teacher", "--epochs", "15")
Invoke-Stage "02_baseline"  @("-m", "q1_kd.train_baseline", "--arch", "all", "--epochs", "30")
Invoke-Stage "03_kd"        @("-m", "q1_kd.train_kd", "--arch", "all", "--epochs", "30")

# ---------------- Question 2: hardware-aware NAS (CIFAR-10) ----------------
Invoke-Stage "04_nas_search" @("-m", "q2_nas.search_optuna", "--trials", "40", "--epochs", "5")
Invoke-Stage "05_nas_train"  @("-m", "q2_nas.train_best", "--epochs", "50")

# ---------------- Deployment, inference and benchmarking ----------------
Invoke-Stage "06_export_onnx" @("-m", "inference.export_onnx", "--all")
Invoke-Stage "07_infer_torch" @("-m", "inference.infer", "--all", "--testset")
Invoke-Stage "08_infer_onnx"  @("-m", "inference.infer_onnx", "--all", "--testset")
Invoke-Stage "09_benchmarks"  @("-m", "benchmark.run_benchmarks")
Invoke-Stage "10_plots"       @("-m", "benchmark.make_plots")
Write-Host "`nAll stages finished." -ForegroundColor Green
