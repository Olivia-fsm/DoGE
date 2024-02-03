#!/bin/bash
pip install -r requirements.txt

export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
python src/run.py --config_json config/doge.json --wandb_proj doge --wandb_run DOGE-82M --total_iterations 10000
