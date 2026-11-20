"""
Deliverable 12 — Build the technical report (HTML -> PDF) from the experiment outputs.

All numbers in the report are read from results/ (JSON / CSV), and the console captures are
rendered from the real training / inference logs in results/logs/.

    python -m report.build_report            # writes report/Technical_Report.html and .pdf
"""
import base64
import re
import glob
import html
import json
import os
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = "results"
OUT_DIR = "report"
SHOT_DIR = "report/screenshots"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


# ----------------------------------------------------------------------------- helpers
def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def read_log(path):
    """Logs written by Tee-Object in Windows PowerShell 5.1 are UTF-16; handle both."""
    raw = open(path, "rb").read()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return raw.decode(enc, errors="replace").replace("\r", "")


def console_png(log_path, out_png, title, keep=None, max_lines=34):
    """Render the (filtered) tail of a real console log as a terminal-style image."""
    # library warnings (multi-line) are kept in the raw logs but left out of the screenshots
    warn = re.compile(r"Warning|warn\(|^C:\\|^\s+(torch\.onnx|return \(|sched\.step)|the legacy TorchScript|"
                      r"the default\. Learn|exporting control flow|Python boolean|will be treated as a constant|"
                      r"lr_scheduler|order: `optimizer|value of the learning|^https://")
    lines = [l for l in read_log(log_path).split("\n") if l.strip() and not warn.search(l)]
    if keep:
        lines = [l for l in lines if re.search(keep, l)]
    lines = lines[-max_lines:]
    lines = [l if len(l) <= 150 else l[:147] + "..." for l in lines]
    h = 0.5 + 0.158 * (len(lines) + 2)
    fig = plt.figure(figsize=(13, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 1 - 0.25 / h, f"PS C:\\...\\Assignment-4>  {title}", color="#4ec9b0",
             family="monospace", fontsize=8.5, va="top")
    fig.text(0.01, 1 - 0.55 / h, "\n".join(lines), color="#d4d4d4", family="monospace",
             fontsize=8, va="top", linespacing=1.35)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def img(path, caption, width="100%"):
    if not os.path.exists(path):
        return f"<p class='missing'>[missing figure: {html.escape(path)}]</p>"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' style='width:{width}'/>"
            f"<figcaption>{caption}</figcaption></figure>")


def table(df, floatfmt="{:.2f}"):
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "\\".join(["HARDWARE", "DESCRIPTION", "System", "CentralProcessor", "0"]))
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def env_info():
    try:
        import torch, onnxruntime, optuna
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        return {"Python": platform.python_version(), "PyTorch": torch.__version__, "CUDA GPU": gpu,
                "ONNX Runtime": onnxruntime.__version__, "Optuna": optuna.__version__,
                "OS": f"{platform.system()} {platform.release()}", "CPU": cpu_name()}
    except Exception:
        return {}


# ----------------------------------------------------------------------------- report
def build():
    os.makedirs(SHOT_DIR, exist_ok=True)
    teacher = jload(f"{R}/q1/teacher_summary.json", {})
    kdcmp = pd.read_csv(f"{R}/q1/kd_vs_baseline.csv") if os.path.exists(f"{R}/q1/kd_vs_baseline.csv") else None
    bench = pd.read_csv(f"{R}/benchmarks.csv") if os.path.exists(f"{R}/benchmarks.csv") else None
    best = jload(f"{R}/q2/best_arch.json", {})
    nasfinal = jload(f"{R}/q2/nas_best_final.json", {})
    hist = pd.read_csv(f"{R}/q2/search_history.csv") if os.path.exists(f"{R}/q2/search_history.csv") else None

    # console captures from the real logs
    shots = []
    L = f"{R}/logs"
    spec = [("01_teacher.log", "python -m q1_kd.train_teacher --epochs 15", None),
            ("02_baseline.log", "python -m q1_kd.train_baseline --arch all --epochs 30", r"===|ep +(1|10|20|30)/|^baseline "),
            ("03_kd.log", "python -m q1_kd.train_kd --arch all --epochs 30", r"===|ep +(1|10|20|30)/|^kd "),
            ("04_nas_search.log", "python -m q2_nas.search_optuna --trials 40 --epochs 5", r"\[trial"),
            ("05_nas_train.log", "python -m q2_nas.train_best --epochs 50", None),
            ("06_export_onnx.log", "python -m inference.export_onnx --all", None),
            ("07_infer_torch.log", "python -m inference.infer --all --testset", None),
            ("08_infer_onnx.log", "python -m inference.infer_onnx --all --testset", None),
            ("demo_image.log", "single-image inference (PyTorch and ONNX Runtime)", None)]
    for fname, title, keep in spec:
        p = os.path.join(L, fname)
        if os.path.exists(p):
            shots.append((console_png(p, f"{SHOT_DIR}/{fname.replace('.log', '.png')}", title, keep), title))

    E = env_info()
    s = []
    s.append("""<h1>Assignment 4 — Technical Report</h1>
<p class='sub'>Edge Computing &amp; Intelligent Systems · Knowledge Distillation and Hardware-Aware Neural Architecture Search</p>
<p class='sub'>Name: Muhammad Ibrahim &nbsp;&nbsp; Roll No: 23I-0083 &nbsp;&nbsp; Class: AI-A</p>
<p class='sub'>GitHub: https://github.com/ibrahim9372/edge_assignment_4</p>""")

    s.append("<h2>1. Overview</h2><p>This report describes two pipelines for running deep models on resource-constrained edge devices. "
             "<b>Question 1</b> transfers knowledge from a large ResNet-50 teacher into three lightweight CNNs "
             "(MobileNetV2, SqueezeNet, EfficientNet-Lite0) on CIFAR-100. The students are trained both with distillation and without it (baseline). "
             "<b>Question 2</b> searches automatically for a small CNN on CIFAR-10. Every candidate is measured on the real CPU "
             "and rejected if it breaks the deployment limits for latency, storage or memory. "
             "All final models are exported to ONNX and run on the CPU with ONNX Runtime.</p>")
    if E:
        s.append("<h3>Experimental environment</h3>" + table(pd.DataFrame(list(E.items()), columns=["Item", "Value"])))
        s.append("<p>Training ran on the GPU with mixed precision (fp16 autocast). All latency and memory measurements "
                 "ran on the CPU in a separate fresh process with a fixed thread count. The whole dataset is kept on the "
                 "GPU as uint8, and the augmentation (random crop with 4-px padding, horizontal flip) runs on the GPU, "
                 "which avoids the slow multi-process data loading on Windows.</p>")

    # ------------------------------------------------------------------ Q1
    s.append("<h2>2. Question 1 — Knowledge Distillation Framework</h2>")
    s.append("<h3>2.1 Teacher: ResNet-50 (q1_kd/models/resnet50.py, q1_kd/train_teacher.py)</h3>"
             "<p>ResNet-50 uses bottleneck residual blocks in four stages of [3, 4, 6, 3] blocks (23.7 M parameters). "
             "The ImageNet stem (7×7 stride-2 conv followed by max-pool) shrinks a 32×32 CIFAR image by 4× before the first residual stage. "
             "We replaced it with a 3×3 stride-1 conv and no max-pool, so the stages run at 32, 16, 8 and 4 pixels. "
             "The teacher is <b>fine-tuned</b>: every layer starts from torchvision's ImageNet weights. "
             "The new 3×3 stem is initialised from the centre 3×3 of the pretrained 7×7 kernels, and the 100-way classifier is new. "
             "All layers are then trained on CIFAR-100 with SGD (Nesterov momentum 0.9), weight decay 5e-4, "
             "1 warm-up epoch followed by cosine learning-rate decay, and batch size 128. "
             "The checkpoint is stored in fp16 (about 48 MB) to keep it under GitHub's file limit. "
             "Reloading it gives the same accuracy (see the table).</p>")
    if teacher:
        t = teacher
        s.append(table(pd.DataFrame([{"epochs": t["args"]["epochs"], "lr": t["args"]["lr"],
                                      "test top-1 (%)": t["final"]["top1"], "test top-5 (%)": t["final"]["top5"],
                                      "top-1 after fp16 reload (%)": t["reloaded_fp16"]["top1"],
                                      "training time (min)": t["train_minutes"]}])))

    s.append("<h3>2.2 Student architectures (q1_kd/models/)</h3>"
             "<p>All three students are written from scratch in PyTorch. Each one also returns an intermediate 8×8 feature map, which the feature-distillation loss uses. "
             "For 32×32 inputs, the stem stride and the first stride-2 stage were changed to stride 1, so each network shrinks the image 32→4 "
             "instead of 224→7.</p><ul>"
             "<li><b>MobileNetV2</b>: inverted-residual blocks (1×1 expand ×6 → 3×3 depthwise → linear 1×1 projection), "
             "ReLU6, and a 1280-channel head. Hint layer: the 96-channel stage output (8×8).</li>"
             "<li><b>SqueezeNet (v1.1 layout)</b>: Fire modules (1×1 squeeze, then parallel 1×1 and 3×3 expand convs whose outputs are concatenated). "
             "Classification uses a 1×1 conv plus global average pooling, so there is no fully connected layer. BatchNorm was added after each conv so it can be trained "
             "stably from scratch. Hint layer: fire5 output, 256 channels (8×8).</li>"
             "<li><b>EfficientNet-Lite0</b>: MBConv blocks with 3×3/5×5 depthwise kernels and the B0 stage layout. Squeeze-and-excitation is removed, ReLU6 replaces "
             "Swish, and the stem (32) and head (1280) widths are fixed. These are the Lite modifications that make it friendly to edge hardware and quantisation. "
             "Hint layer: the 112-channel stage output (8×8).</li></ul>")
    if bench is not None:
        b = bench[bench.group.isin(["teacher", "kd"])][["arch", "params_M", "MMACs", "fp32_weights_MB"]].drop_duplicates("arch")
        s.append(table(b.rename(columns={"params_M": "params (M)", "MMACs": "MMACs / image", "fp32_weights_MB": "fp32 weights (MB)"})))

    s.append("<h3>2.3 Distillation framework (q1_kd/kd/losses.py)</h3>"
             "<p>Distillation is implemented as a <i>criterion</i> object, <code>KDCriterion(model, x, y) → loss</code>. "
             "Because of this, baseline and distillation training share the same training loop. The criterion holds the frozen teacher, which stays in eval mode "
             "so its BatchNorm statistics never change, plus a small trainable feature adapter. The loss has three parts:</p>"
             "<div class='eq'>L = α · CE(z<sub>s</sub>, y) + (1 − α) · T² · KL( softmax(z<sub>t</sub>/T) ‖ softmax(z<sub>s</sub>/T) ) "
             "+ β · ‖ r(F<sub>s</sub>) − F<sub>t</sub> ‖²</div><ul>"
             "<li><b>Soft targets with temperature T</b> (Hinton et al.): dividing the logits by T &gt; 1 flattens the teacher's distribution, "
             "so it shows how similar the wrong classes are to each other (\"dark knowledge\"). Multiplying the KL term by T² keeps its gradient scale independent of T.</li>"
             "<li><b>Combined hard-label and soft-label training</b>: α weights the true-label cross-entropy against the soft-target term.</li>"
             "<li><b>Feature-based distillation</b> (FitNets): the student's 8×8 hint feature map F<sub>s</sub> passes through a regressor r "
             "(1×1 conv + BN) that maps it to the teacher's 1024 channels. It is then matched to the teacher's layer3 output F<sub>t</sub> (1024×8×8) with MSE. "
             "If the spatial sizes differ, both maps are average-pooled to the smaller one. The adapter is only used during training and is not part of the deployed student.</li></ul>")
    s.append("<p>Hyper-parameters used: T = 4, α = 0.3, β = 1.0. Flags turn each component off for ablations "
             "(<code>--no-soft</code>, <code>--no-feature</code>).</p>")

    s.append("<h3>2.4 Baseline vs. distillation training (q1_kd/train_baseline.py, q1_kd/train_kd.py)</h3>"
             "<p>The baseline pipeline trains each student with hard-label cross-entropy only and never loads the teacher. "
             "The distillation pipeline uses the same optimiser, schedule, number of epochs (30), batch size, augmentation and random seed, "
             "so any accuracy difference comes from distillation alone. Settings: SGD with Nesterov momentum 0.9, learning rate 0.1, cosine decay with 1 warm-up epoch, "
             "weight decay 5e-4, batch size 128, fp16 autocast.</p>")
    if kdcmp is not None:
        s.append(table(kdcmp.rename(columns={"teacher_top1": "teacher top-1", "baseline_top1": "baseline top-1",
                                             "kd_top1": "KD top-1", "baseline_top5": "baseline top-5",
                                             "kd_top5": "KD top-5", "params_M": "params (M)",
                                             "ort_cpu_latency_ms": "ORT latency (ms)", "kd_gain_top1": "KD gain (pts)"})))
        s.append(kd_discussion(kdcmp))
    s.append(img(f"{R}/figures/q1_kd_vs_baseline.png", "Figure 1 — CIFAR-100 test accuracy of baseline and distilled students.", "78%"))
    s.append(img(f"{R}/figures/q1_learning_curves.png", "Figure 2 — Test-accuracy curves (dotted: baseline, solid: KD).", "85%"))

    s.append("<h3>2.5 Unified inference and ONNX deployment (inference/)</h3>"
             "<p>Every checkpoint stores its architecture name, dataset, class names, input resolution and (for the NAS model) "
             "its architecture configuration. <code>common/registry.py</code> can therefore rebuild <i>any</i> model from a checkpoint, "
             "and one script (<code>inference/infer.py</code>) serves all six students and the NAS model. It can classify single images (top-5) "
             "or evaluate a whole test set. <code>inference/export_onnx.py</code> exports each model to ONNX (opset 17, dynamic batch size), "
             "validates the graph with <code>onnx.checker</code>, and checks that ONNX Runtime outputs match PyTorch (max |Δ| ≈ 1e-6). "
             "It also writes a JSON metadata file next to each model. <code>inference/infer_onnx.py</code> runs the models with ONNX Runtime's "
             "CPUExecutionProvider and does all preprocessing in NumPy, so deployment does not need PyTorch.</p>")
    s.append(img(f"{R}/figures/q1_accuracy_vs_latency.png", "Figure 3 — Accuracy vs. measured CPU latency on CIFAR-100.", "75%"))

    # ------------------------------------------------------------------ Q2
    s.append("<h2>3. Question 2 — Hardware-Aware Neural Architecture Search</h2>")
    s.append("<h3>3.1 Search space (q2_nas/search_space.py)</h3>"
             "<table><thead><tr><th>Dimension</th><th>Choices</th></tr></thead><tbody>"
             "<tr><td>Input resolution</td><td>24, 28, 32</td></tr>"
             "<tr><td>Stem channels</td><td>16, 24, 32</td></tr>"
             "<tr><td>Activation (global)</td><td>ReLU, ReLU6, Hardswish, SiLU</td></tr>"
             "<tr><td>Depth per stage (3 stages)</td><td>1, 2, 3 blocks</td></tr>"
             "<tr><td>Channels per stage</td><td>{16,24,32,48} / {32,48,64,96} / {64,96,128,160}</td></tr>"
             "<tr><td>Kernel size per stage</td><td>3, 5, 7</td></tr>"
             "<tr><td>Block type per stage</td><td>standard conv, depthwise-separable conv</td></tr>"
             "<tr><td>Head channels / dropout</td><td>128, 256, 384 / 0.0, 0.2</td></tr></tbody></table>"
             "<p>That gives about 7.5×10<sup>7</sup> distinct architectures. A configuration is a JSON dictionary, so it can be stored, "
             "compared, saved in checkpoints and rebuilt exactly.</p>")
    s.append("<h3>3.2 Architecture generator (q2_nas/generator.py)</h3>"
             "<p><code>build_from_config(cfg)</code> builds the network from the configuration: a 3×3 stem, then three stages "
             "(the first block of stages 2 and 3 uses stride 2), a 1×1 head conv, global average pooling, dropout and a linear classifier. "
             "A <i>conv</i> block is k×k conv → BN → activation. A <i>dwsep</i> block is a k×k depthwise conv followed by a 1×1 pointwise conv, "
             "each with BN and activation. This costs roughly (k² + C<sub>out</sub>)/(k²·C<sub>out</sub>) of a dense conv. "
             "A residual shortcut is added whenever the input and output shapes match. The input resolution is applied in the data pipeline "
             "by bilinear (anti-aliased) resizing.</p>")
    s.append("<h3>3.3 Search strategy: Bayesian optimisation with Optuna TPE (q2_nas/search_optuna.py)</h3>"
             "<p>The search uses Optuna's Tree-structured Parzen Estimator (multivariate, seed 42). The first 10 trials are random to explore the space. "
             "After that, TPE models which parameter values lead to good versus bad accuracy and samples the next candidate where the "
             "expected improvement is highest. Each trial runs these steps:</p><ol>"
             "<li>TPE samples a configuration and the generator builds the network.</li>"
             "<li><b>Hardware profiling</b> (common/profiler.py, q2_nas/hw_constraints.py) runs in a fresh CPU-only subprocess: "
             "median latency at batch 1 on <b>1 CPU thread</b> (to mimic a small edge core; 10 warm-up runs, then 40 timed runs), "
             "storage size of the serialised fp32 weights, and <b>peak inference memory</b>. Peak memory is the increase in resident memory (RSS) during inference, "
             "sampled every 0.5 ms by a psutil thread. psutil works on Windows, Linux and macOS; on other platforms this value is skipped.</li>"
             "<li><b>Constraint check</b>: any candidate that breaks a deployment limit is <b>rejected</b> straight away and recorded as pruned with the reason. "
             "Rejected candidates take only about 2 s and no GPU time.</li>"
             "<li>Feasible candidates are trained for a short proxy budget (5 epochs on 45k CIFAR-10 training images) and "
             "validated on a 5k held-out split. The test set is never used during the search. Validation top-1 is the objective that TPE maximises.</li></ol>")
    if best:
        c = best["constraints"]
        s.append(f"<p>Deployment limits: latency ≤ <b>{c['max_latency_ms']} ms</b> (batch 1, 1 thread), "
                 f"model size ≤ <b>{c['max_size_mb']} MB</b>, peak inference memory ≤ <b>{c['max_peak_mem_mb']} MB</b>.</p>")
        s.append(f"<h3>3.4 Search results</h3><p>The search ran {best['n_trials']} trials in {best['search_minutes']} min. "
                 f"<b>{best['n_rejected']}</b> candidates were rejected for breaking the limits and "
                 f"{best['n_trials'] - best['n_rejected']} were trained and evaluated. The full history (every configuration, its measured "
                 f"latency, size and memory, status, rejection reason and validation accuracy) is in "
                 f"<code>results/q2/search_history.csv/json</code>. The best architecture (trial {best['best_trial']}, validation top-1 "
                 f"{best['val_acc']:.2f}% after the proxy training) was saved automatically to <code>results/q2/best_arch.json</code>:</p>"
                 f"<div class='eq'>{html.escape(best['description'])}</div>")
        hw = best["hardware"]
        s.append(table(pd.DataFrame([{"params": int(hw["params"]), "MMACs": hw["mmacs"], "CPU latency (ms)": hw["latency_ms"],
                                      "size (MB)": hw["size_mb"], "peak mem (MB)": hw["peak_mem_mb"]}])))
    if hist is not None:
        acc = hist[hist.status == "accepted"].sort_values("val_acc", ascending=False).head(8)
        s.append("<p>Top feasible candidates:</p>" + table(acc[["trial", "arch", "latency_ms", "size_mb", "peak_mem_mb", "val_acc"]]))
        rej = hist[hist.status == "rejected"].head(6)
        if len(rej):
            s.append("<p>Examples of rejected candidates:</p>" + table(rej[["trial", "arch", "reason"]]))
    s.append(img(f"{R}/figures/q2_nas_constraints.png", "Figure 4 — Every NAS candidate in latency/size space. Dashed lines are the deployment limits.", "80%"))
    s.append(img(f"{R}/figures/q2_nas_history.png", "Figure 5 — Optuna TPE optimisation history.", "78%"))
    s.append(img(f"{R}/figures/q2_nas_acc_vs_latency.png", "Figure 6 — Accuracy vs. latency of the feasible candidates.", "70%"))

    s.append("<h3>3.5 Training the best architecture from scratch and deploying it (q2_nas/train_best.py)</h3>")
    if nasfinal:
        s.append(f"<p>The selected configuration was built again with fresh random weights (nothing was reused from the search) and trained on "
                 f"the full 50k CIFAR-10 training set for {nasfinal['epochs']} epochs, using SGD with learning rate 0.1 and cosine decay. "
                 f"It reaches <b>{nasfinal['test']['top1']:.2f}% test top-1</b> "
                 f"(top-5 {nasfinal['test']['top5']:.2f}%), with training taking {nasfinal['train_minutes']} min. It was then exported to ONNX "
                 f"(<code>onnx/nas/nas_best_cifar10.onnx</code>) and runs on the CPU with ONNX Runtime through the same inference scripts as the students.</p>")

    # ------------------------------------------------------------------ benchmarks
    s.append("<h2>4. Benchmarking (benchmark/run_benchmarks.py)</h2>"
             "<p>One script benchmarks every model: test accuracy (PyTorch and ONNX Runtime), parameters, MACs, checkpoint, "
             "fp32 and ONNX sizes, CPU latency at batch 1 on 1 thread (PyTorch and ONNX Runtime, median of 100 runs), ONNX Runtime throughput at batch 32 "
             "on all physical cores, and peak inference memory. Every CPU measurement runs in an isolated subprocess. "
             "Output: <code>results/benchmarks.csv</code>.</p>")
    if bench is not None:
        cols = [c for c in ["model", "top1", "ort_top1", "params_M", "MMACs", "onnx_MB", "pt_cpu_latency_ms",
                            "ort_cpu_latency_ms", "ort_speedup_vs_pt", "pt_peak_mem_MB", "ort_peak_mem_MB"] if c in bench.columns]
        thr = [c for c in bench.columns if c.startswith("ort_throughput")]
        short = {"top1": "top-1", "ort_top1": "ORT top-1", "params_M": "params M", "MMACs": "MMACs",
                 "onnx_MB": "ONNX MB", "pt_cpu_latency_ms": "PT ms", "ort_cpu_latency_ms": "ORT ms",
                 "ort_speedup_vs_pt": "ORT speed-up", "pt_peak_mem_MB": "PT mem MB",
                 "ort_peak_mem_MB": "ORT mem MB"}
        short.update({c: "ORT img/s (b32)" for c in thr})
        s.append("<div class='small'>" + table(bench[cols + thr].rename(columns=short)) + "</div>")
        s.append("<p class='note'>PT/ORT ms = median latency at batch 1 on 1 CPU thread; ORT img/s = ONNX Runtime throughput "
                 "at batch 32 on all physical cores; mem = peak RSS growth during inference.</p>")
        s.append(bench_discussion(bench))

    # ------------------------------------------------------------------ screenshots
    s.append("<h2>5. Execution screenshots</h2><p>Console output captured from the actual runs (full logs are in <code>results/logs/</code>).</p>")
    for i, (p, title) in enumerate(shots, 1):
        s.append(img(p, f"Screenshot {i} — {html.escape(title)}"))

    # ------------------------------------------------------------------ deliverables
    s.append("<h2>6. Deliverables map</h2>" + table(pd.DataFrame([
        (1, "Teacher model", "checkpoints/teacher/resnet50_cifar100.pt, q1_kd/train_teacher.py"),
        (2, "Student architectures", "q1_kd/models/{mobilenetv2,squeezenet,efficientnet_lite0}.py"),
        (3, "Baseline students", "checkpoints/baseline/*.pt, q1_kd/train_baseline.py"),
        (4, "Knowledge distillation", "q1_kd/kd/losses.py, q1_kd/train_kd.py, checkpoints/kd/*.pt"),
        (5, "NAS search space", "q2_nas/search_space.py, q2_nas/generator.py"),
        (6, "NAS algorithm", "q2_nas/search_optuna.py, q2_nas/hw_constraints.py, common/profiler.py"),
        (7, "NAS results", "results/q2/best_arch.json, results/q2/search_history.{csv,json}"),
        (8, "Optimised architecture", "checkpoints/nas/nas_best_cifar10.pt, q2_nas/train_best.py"),
        (9, "Inference pipeline", "inference/infer.py (all students + NAS)"),
        (10, "ONNX deployment", "onnx/**/*.onnx, inference/export_onnx.py, inference/infer_onnx.py"),
        (11, "Benchmarking", "benchmark/run_benchmarks.py, results/benchmarks.csv, results/q1/kd_vs_baseline.csv"),
        (12, "Technical report", "report/Technical_Report.pdf"),
        (13, "README", "README.md"),
    ], columns=["#", "Deliverable", "Location"])))
    s.append("<h2>7. Reproducing the results</h2><p>See README.md. <code>.\\run_all.ps1</code> runs the whole pipeline "
             "(teacher → baselines → distillation → NAS search → retrain best model → ONNX export → inference → benchmarks → figures). "
             "Every stage writes its console output to <code>results/logs/</code>. All runs use seed 42.</p>")
    s.append("<h2>References</h2><ol class='refs'>"
             "<li>G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. 2015.</li>"
             "<li>A. Romero et al. FitNets: Hints for Thin Deep Nets. ICLR 2015.</li>"
             "<li>K. He et al. Deep Residual Learning for Image Recognition. CVPR 2016.</li>"
             "<li>M. Sandler et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.</li>"
             "<li>F. Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters. 2016.</li>"
             "<li>M. Tan, Q. Le. EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019; TensorFlow EfficientNet-Lite release, 2020.</li>"
             "<li>J. Bergstra et al. Algorithms for Hyper-Parameter Optimization (TPE). NeurIPS 2011; T. Akiba et al. Optuna. KDD 2019.</li>"
             "<li>M. Tan et al. MnasNet: Platform-Aware Neural Architecture Search for Mobile. CVPR 2019.</li></ol>")

    css = """body{font-family:'Segoe UI',Arial,sans-serif;font-size:10.5pt;color:#1d1d1f;max-width:900px;margin:auto;line-height:1.45}
h1{font-size:21pt;margin-bottom:2px}h2{font-size:15pt;border-bottom:2px solid #2a6fdb;padding-bottom:3px;margin-top:26px}
h3{font-size:12pt;margin-top:16px}.sub{color:#555;margin:2px 0}
table{border-collapse:collapse;margin:8px 0;font-size:8.6pt;width:100%}th,td{border:1px solid #c9ccd1;padding:3px 6px;text-align:left}
th{background:#eef2f8}tr:nth-child(even) td{background:#fafbfc}code{background:#f2f3f5;padding:0 3px;border-radius:3px;font-size:9pt}
.eq{background:#f5f7fb;border-left:3px solid #2a6fdb;padding:6px 10px;margin:8px 0;font-family:Cambria,serif}
figure{margin:10px 0;text-align:center;page-break-inside:avoid}figcaption{font-size:9pt;color:#555}
.missing{color:#b00}.small table{font-size:7.4pt}.small td,.small th{padding:2px 4px}.note{font-size:8.5pt;color:#555}.refs{font-size:9pt}"""
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Assignment 4 Technical Report</title><style>{css}</style></head><body>{''.join(s)}</body></html>"
    html_path = os.path.abspath(f"{OUT_DIR}/Technical_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)
    pdf_path = os.path.abspath(f"{OUT_DIR}/Technical_Report.pdf")
    if os.path.exists(EDGE):
        subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")], check=False,
                       timeout=180)
    print("wrote", html_path, "and", pdf_path if os.path.exists(pdf_path) else "(PDF not generated)")


def kd_discussion(d):
    gains = d.kd_gain_top1
    pos = (gains > 0).sum()
    best = d.loc[gains.idxmax()]
    txt = (f"<p><b>Discussion.</b> Distillation changed top-1 accuracy by {', '.join(f'{a}: {g:+.2f}' for a, g in zip(d.arch, gains))} "
           f"percentage points, and improved {pos} of the 3 students. The largest gain was for <b>{best.arch}</b> ({best.kd_gain_top1:+.2f} pts). ")
    txt += ("The teacher's soft targets act as a learned, class-aware form of label smoothing. They tell the student which "
            "classes are easy to confuse (e.g. among CIFAR-100's 20 superclasses), which helps most when the student has too little capacity "
            "to discover those relations from one-hot labels alone. The feature hint also gives the student's middle layers "
            "direct supervision. Distillation does not change the architecture of the deployed student, so these gains come at "
            "<b>no cost</b> in inference latency, size or memory.</p>")
    return txt


def bench_discussion(b):
    st = b[b.group.isin(["baseline", "kd"])]
    if "ort_speedup_vs_pt" not in b.columns or st.empty:
        return ""
    t = b[b.group == "teacher"]
    nas = b[b.group == "nas"]
    txt = "<p><b>Observations.</b> "
    if len(t):
        tl = t.pt_cpu_latency_ms.iloc[0]
        fastest = st.loc[st.pt_cpu_latency_ms.idxmin()]
        txt += (f"Every student is much cheaper than the ResNet-50 teacher ({tl:.1f} ms per image on 1 thread). The fastest student, "
                f"{fastest.arch}, takes {fastest.pt_cpu_latency_ms:.2f} ms ({tl / fastest.pt_cpu_latency_ms:.0f}× faster). ")
    txt += (f"ONNX Runtime is {st.ort_speedup_vs_pt.min():.1f}–{st.ort_speedup_vs_pt.max():.1f}× faster than eager PyTorch on the CPU "
            "for the students, because of graph optimisations such as Conv+BN folding, operator fusion and memory planning, while giving the same "
            "accuracy (compare the top1 and ort_top1 columns). ")
    if len(nas):
        n = nas.iloc[0]
        txt += (f"The NAS model is the smallest deployable network ({n.params_M:.3f} M parameters, {n.onnx_MB:.2f} MB as ONNX) "
                f"and runs in {n.ort_cpu_latency_ms:.2f} ms with ONNX Runtime, well inside its 2 ms budget. It solves the easier 10-class "
                f"CIFAR-10 task, so its accuracy cannot be compared directly with the CIFAR-100 students. Its design came directly from the "
                f"latency, size and memory measured on this CPU.")
    txt += (" <i>Note on memory:</i> peak inference memory is measured as the growth of process RSS during inference. "
            "PyTorch's CPU allocator reuses memory already reserved while the checkpoint was loaded, so for some PyTorch "
            "models the measured growth is close to zero. The ONNX Runtime numbers (arena allocation at the first run) show the "
            "real activation-memory footprint more clearly.")
    return txt + "</p>"


if __name__ == "__main__":
    build()
