# MP1 submission code

This submission directory contains the implementation and supplied WikiText-2
data under `code/` for the frozen serial two-student predictor. The matching
checkpoint is `checkpoint_bundle/checkpoint.pt`. The submitted coursework
report is `report.pdf`, with `report.docx` retained as its editable source.
Primary records for the report's early experiment examples are in
`evidence/experiments/early-exploration/manifest.json`.

## Install and score

Use Python 3.12. From the submission repository root, enter `code/`, create a
virtual environment, and install PyTorch 2.7.1 and the other pinned
dependencies:

```bash
cd code
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate.py --checkpoint ../checkpoint_bundle/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output ../recheck/test_cpu_fp32.json
```

The last command scores the complete test split without retraining. Read the
`bpb` field in `../recheck/test_cpu_fp32.json`. The recorded result from three
independent CPU FP32 runs is **1.4969323726595785 BPB**. The checkpoint
SHA-256 is
`a46036ff367b26b081a253213f367657e7d65a10cd5fdd537b72c8379b1fe649`.
The measurement record is in `../evidence/candidate/resource_summary.json`:
three local candidate/baseline time ratios are 4.801, 4.622 and 4.574; median
time ratio is 4.646; peak process RSS is 1.618 GiB; and the complete
uncompressed inference asset set is 28.772 MiB. Reference-machine timing and
resource compliance still require verification.

`common.py`, `evaluate.py`, the tokenizer and the dataset are the unchanged
course files. The checkpoint selects `temperature_gated_ensemble2.py`, which
loads `gated_ensemble2.py`, `ensemble2.py` and `model.py`. Student 2 runs
serially only for windows selected by a causal confidence gate. There are no
additional Python worker threads.

## Training recipe and provenance

Both width-256 students were trained from random initialization using only the
supplied training split. The training-only teacher checkpoint is at
`runs/temperature-3/t105/checkpoint.pt` (SHA-256
`dc19ace484134e6700eba49981fb1a8f8457adfc6c1a5f4aff38d630016505d5`).
It is not loaded by the frozen inference model and is not part of the inference
asset count. Its selection record is in `../evidence/teacher/`.

The exact 14,400-step student settings are in `run_distillation_14400.ps1` and
`../evidence/experiments/distillation-14400/experiment_plan.json`. On Windows,
from the `code/` directory, prepare `.venv` for CPU PyTorch and `.venv-gpu` for
CUDA PyTorch, then run the commands below from that directory. The CUDA
environment used for these experiments had PyTorch 2.7.1+cu128; use a
compatible NVIDIA driver and install it with:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
```

The training and packaging commands are:

```powershell
.\run_distillation_14400.ps1 -Name kd075-s17 -SoftWeight 0.75 -Seed 17
.\run_distillation_14400.ps1 -Name kd075-s23 -SoftWeight 0.75 -Seed 23
.\.venv\Scripts\python.exe package_pair.py --first runs/distillation-14400/kd075-s17/ema07.pt --second runs/distillation-14400/kd075-s23/ema07.pt --output runs/distillation-14400/dual-kd075/checkpoint.pt
.\.venv\Scripts\python.exe package_temperature_gated_pair.py --checkpoint runs/distillation-14400/dual-kd075/checkpoint.pt --output runs/distillation-14400/gated-t0975/checkpoint.pt --temperature-first 0.975 --temperature-second 0.975 --threshold 0.16
```

`train_distilled.py`, `average_checkpoints.py`, `package_pair.py`, and
`package_temperature_gated_pair.py` implement those steps. The teacher and
student settings, validation selection, checkpoint lineage, training costs,
and comparison results are recorded in `../evidence/experiments/`. The added
two student runs processed 235,929,600 training targets in 1,062.99 GPU
seconds. Across the eight recorded student runs, the cumulative student total
was 943,718,400 training targets and 3,979.42 GPU seconds; teacher training
and search are additional. The final temperatures and gate threshold were
selected on validation before the final checkpoint was tested.

The baseline is `model.py` with `configs/baseline.json`; `train.py` reproduces
it. Baseline measurements are in `../evidence/baseline/`. Comparison and
ablation results used in development are recorded in `../evidence/experiments/`.

## Reuse and AI assistance

The original GPT baseline, trainer, fixed data, tokenizer and scorer are the
course-provided implementation. RMSNorm and GEGLU experiments adapt published
methods. With AI assistance, I implemented reusable scripts for experiment orchestration, 
checkpoint averaging, packaging, automated testing, resource measurement, and result visualization, 
and improved the documentation. All training used the supplied training text;
the frozen predictor does not access external services.
