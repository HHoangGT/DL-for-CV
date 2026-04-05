#!/usr/bin/env bash
set -e
python -m src.train --config configs/deeplabv3plus_resnet50.yaml
python -m src.train --config configs/deeplabv3plus_convnext_tiny.yaml
python -m src.train --config configs/deeplabv3plus_mit_b0.yaml
python -m src.summarize_results --experiments_dir artifacts/experiments
