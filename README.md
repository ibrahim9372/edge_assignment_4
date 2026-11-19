# Edge Computing & Intelligent Systems — Assignment 4

**Q1:** Knowledge distillation from a ResNet-50 teacher into MobileNetV2, SqueezeNet and EfficientNet-Lite0 (CIFAR-100).
**Q2:** Hardware-aware neural architecture search (Optuna TPE) for CPU edge deployment (CIFAR-10).
Both questions end with ONNX export and CPU inference through ONNX Runtime.

## Repository layout

```
common/            shared code
  data.py            CIFAR loading, GPU-resident data + GPU augmentation
  train_loop.py      generic AMP training / evaluation loop (criterion-pluggable)
  profiler.py        CPU latency / storage size / peak inference memory (isolated subprocess)
  registry.py        build any model from a name/config; load any checkpoint
  utils.py           seeding, checkpoints, CSV logging, params / MAC counting
q1_kd/             Question 1
  models/            resnet50.py, mobilenetv2.py, squeezenet.py, efficientnet_lite0.py
  kd/losses.py       KD criterion: hard CE + temperature soft targets + feature (hint) loss
  train_teacher.py   ResNet-50 teacher (ImageNet-initialised, fine-tuned on CIFAR-100)
  train_baseline.py  students without distillation
  train_kd.py        students with distillation
  train_students.py  shared implementation of the two student pipelines
q2_nas/            Question 2
  search_space.py    searchable space (depth, channels, kernel, dw-separable, activation, resolution)
  generator.py       config -> nn.Module
  hw_constraints.py  deployment constraints + hardware-aware rejection
  search_optuna.py   Bayesian optimisation (TPE) search loop
  train_best.py      retrain best architecture from scratch
inference/         unified deployment
  infer.py           PyTorch inference for all students + NAS model
  export_onnx.py     ONNX export (+ numerical check vs PyTorch)
  infer_onnx.py      ONNX Runtime CPU inference
benchmark/
  run_benchmarks.py  accuracy / size / latency / memory for every model -> CSV
  make_plots.py      figures for the report
tools/prepare_data.py  dataset fallback (Hugging Face mirror -> .npz)
checkpoints/       trained models (teacher, baseline, kd, nas)
onnx/              exported ONNX models + metadata sidecars
results/           CSV / JSON results, training logs, figures
report/            technical report (PDF)
run_all.ps1        runs the whole pipeline end to end
```

## Installation

Tested on Windows 11, Python 3.12, NVIDIA RTX 2080 (8 GB), CUDA 12.6 wheels.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Training needs a CUDA GPU. Inference, ONNX export and benchmarking run on the CPU.

### Dataset

The scripts download CIFAR-10/100 automatically through torchvision into `data/`.
If the official server (cs.toronto.edu) is unreachable, use the Hugging Face mirror:

```powershell
# download the 4 parquet files listed in tools/prepare_data.py into data/hf/, then
python -m tools.prepare_data          # writes data/cifar10.npz and data/cifar100.npz
```

## Step-by-step execution

Run every command from the repository root. You can run the whole pipeline with `.\run_all.ps1`, or run the steps one at a time:

| Step | Command | Output |
|---|---|---|
| 1. Teacher | `python -m q1_kd.train_teacher --epochs 30` | `checkpoints/teacher/resnet50_cifar100.pt` |
| 2. Baseline students | `python -m q1_kd.train_baseline --arch all --epochs 40` | `checkpoints/baseline/*.pt` |
| 3. Distilled students | `python -m q1_kd.train_kd --arch all --epochs 40 --T 4 --alpha 0.3 --beta 1` | `checkpoints/kd/*.pt` |
| 4. NAS search | `python -m q2_nas.search_optuna --trials 60 --epochs 6` | `results/q2/search_history.{csv,json}`, `results/q2/best_arch.json` |
| 5. Train best NAS model | `python -m q2_nas.train_best --epochs 80` | `checkpoints/nas/nas_best_cifar10.pt` |
| 6. ONNX export | `python -m inference.export_onnx --all` | `onnx/{baseline,kd,nas}/*.onnx` |
| 7. PyTorch inference | `python -m inference.infer --all --testset` | console |
| 8. ONNX Runtime inference | `python -m inference.infer_onnx --all --testset` | console |
| 9. Benchmarks | `python -m benchmark.run_benchmarks` | `results/benchmarks.csv`, `results/q1/kd_vs_baseline.csv` |
| 10. Figures | `python -m benchmark.make_plots` | `results/figures/*.png` |

Classify a single image with any model:

```powershell
python -m inference.infer      --ckpt checkpoints/kd/mobilenetv2_cifar100.pt --image my_image.png
python -m inference.infer_onnx --onnx onnx/nas/nas_best_cifar10.onnx         --image my_image.png
```

Every script supports `--smoke`, which runs a 1-epoch run on a tiny subset to check the pipeline quickly, and `--help`.

### Useful options

* KD ablations: `--no-soft` (feature loss only), `--no-feature` (soft targets only), `--T`, `--alpha`, `--beta`.
* NAS deployment constraints: `--max-latency-ms`, `--max-size-mb`, `--max-peak-mem-mb`.

## Results

See `results/` and the technical report in `report/`.
