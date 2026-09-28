DASE7506 MP1 frozen predictor bundle

Predictor: serial temperature-calibrated confidence-gated two-student model
Checkpoint: checkpoint.pt
SHA-256: a46036ff367b26b081a253213f367657e7d65a10cd5fdd537b72c8379b1fe649
Implementation module: temperature_gated_ensemble2
Complete-test CPU FP32 BPB: 1.4969323726595785

Use this bundle with the matching sibling code directory. From code/:
python evaluate.py --checkpoint ../checkpoint_bundle/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output ../recheck/test_cpu_fp32.json

The code directory contains the required model modules, evaluator, tokenizer,
data and pinned dependencies. freeze_manifest.json records the validation
selection and hashes of the frozen checkpoint and required source files.
