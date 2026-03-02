# edge-learner

Production-grade edge AI self-learning pipeline for RK3588 NPU.

Continuously improves on-device YOLOv8 detection accuracy through:
- Low-confidence frame collection (Stage 1)
- Two-pass base-model + reference validation (Stages 2-3)
- On-device LoRA retraining with O-LoRA + EWC (Stage 4)
- Evaluation gate + atomic RKNN swap (Stage 5)

Inference code references: https://github.com/airockchip/rknn_model_zoo
RKNN toolkit: https://github.com/airockchip/rknn-toolkit2

## Quick Start

```bash
pip install -r requirements.txt
python -m orchestrator.scheduler
```

## Architecture

```
Camera → RKNN Inference → Confidence Monitor
                              ↓ (low-conf frames)
                         Buffer Manager
                              ↓
                    Base Model Compare (S2)
                              ↓
                   Reference Validator (S3)
                              ↓
                    LoRA Train Loop (S4)
                              ↓
                     Eval Gate (S5a)
                              ↓
                  RKNN Convert + Atomic Swap (S5b)
```

## Structure

```
edge-learner/
├── config/          # YAML configs for pipeline, variants, quant
├── inference/       # RKNN engine (airockchip ref), frame pipeline
├── collector/       # Low-confidence buffer + variant tagger
├── screener/        # Base model compare + reference validator
├── trainer/         # LoRA, O-LoRA, EWC, replay buffer
├── evaluator/       # Metrics, cross-variant test, gate checker
├── deployer/        # LoRA merge, ONNX export, RKNN convert, atomic swap
├── orchestrator/    # Pipeline runner, mode manager, scheduler
├── utils/           # Storage, provenance, metrics logger, cloud sync
└── systemd/         # Service unit files
```

## Supported Hardware
- Rockchip RK3588 / RK3588S (Radxa Rock 5, Orange Pi 5, Edgeble NGC3)
- NPU: 6 TOPS INT8 (3x 2-TOPS cores)

## License
MIT
