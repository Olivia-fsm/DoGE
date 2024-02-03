#!/bin/bash
pip install -r requirements.txt
export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"

# process dataset
python src/data/process_wiki40b.py
python src/run.py --config_json config/doge_ood/wiki40b_catalan.json --wandb_proj doge --wandb_run DOGEood-Catalan --total_iterations 10000
