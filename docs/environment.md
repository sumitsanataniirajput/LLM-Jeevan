# Development environment and training-resource baseline

Hardware was re-detected in this workspace on 2026-10-31, including the Phase 5 multilingual preset. Values are a point-in-time snapshot; rerun `./scripts/hardware.sh --config <preset>.toml` before any real run.

| Resource | Observed |
| --- | --- |
| OS / architecture | Linux x86_64 |
| CPU | Intel Xeon Processor @ 2.60 GHz; 2 logical CPUs |
| RAM | 3.848 GiB visible total; about 3.245 GiB available at inspection |
| CUDA | unavailable; 0 visible devices |
| MPS | unavailable |
| Filesystem | about 20.34 GiB total; 11.12 GiB free on the repository filesystem |
| Python | 3.11.2 in the project `.venv` |
| PyTorch | 2.15.0+cu132; CUDA runtime libraries are present in the wheel but no CUDA device is available |

The current estimate uses FP32 weights, gradients, and AdamW moments, plus a conservative activation/logit estimate and safety margin. The training-FLOP estimate uses a rough 6-parameter-token approximation plus attention matmuls; it is not an exact operation count, measured throughput, or runtime estimate. Reports do not include all Python/runtime/data-cache overhead and are not promises that a run will fit or be fast.

| Preset | Parameters | Configured batch | Conservative batch recommendation | Estimated training memory at configured batch | Approx. checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 132,864 | 8 | 8 | 25 MiB | 1.5 MiB |
| small | 6,852,608 | 4 | 8 | 0.324 GiB | 78.4 MiB |
| medium | 33,674,240 | 2 | 2 | 1.25 GiB | 385.4 MiB |
| multilingual_tiny | 239,360 | 2 | 8 | 11.9 MiB | 2.74 MiB |

For `multilingual_tiny` at batch 2 and 1,000 configured steps, the approximate compute estimate is 417,988,608 FLOPs/step and 417,988,608,000 FLOPs total (~0.418 TFLOP). This is a formula-based estimate only; no throughput or duration is claimed.

Recommendations are capped and advisory. The medium estimate is a memory-planning result only; this two-CPU workspace may train it very slowly. No model is automatically enlarged and no full or real-corpus pretraining was run during development.

Use the project-local virtual environment for dependencies and commands. The system Python is externally managed; do not install project packages into it. The dataset and checkpoint directories contain no real corpus or trained production weights.
