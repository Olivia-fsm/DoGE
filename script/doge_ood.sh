#!/bin/bash
pip install -r requirements.txt
export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"

python src/run.py --config_json config/doge_ood/arxiv_ood.json --wandb_proj doge --wandb_run DOGEood-Arxiv --total_iterations 10000
python src/run.py --config_json config/doge_ood/book_ood.json --wandb_proj doge --wandb_run DOGEood-Book --total_iterations 10000
python src/run.py --config_json config/doge_ood/c4_ood.json --wandb_proj doge --wandb_run DOGEood-C4 --total_iterations 10000
python src/run.py --config_json config/doge_ood/cc_ood.json --wandb_proj doge --wandb_run DOGEood-CC --total_iterations 10000
python src/run.py --config_json config/doge_ood/github_ood.json --wandb_proj doge --wandb_run DOGEood-Github --total_iterations 10000
python src/run.py --config_json config/doge_ood/stack_ood.json --wandb_proj doge --wandb_run DOGEood-Stack --total_iterations 10000
python src/run.py --config_json config/doge_ood/wiki_ood.json --wandb_proj doge --wandb_run DOGEood-Wiki --total_iterations 10000
