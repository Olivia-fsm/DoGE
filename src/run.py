from trainer import *
import logging
from pathlib import Path
import os
import sys
import json
import numpy as np
import argparse
import datasets
import torch
import pickle

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from eval_datasets import get_eval_dataset
from dataloader import DataTrainingArguments, DomainConfigArguments, get_data_collator, get_train_eval_datasets
from models import CausalLMOutputWithDomainIDs, ModelArguments, get_model_from_config, GPT2DoGE
from trainer import FullTrainingArguments

args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--config_json', default='path to json config file', type=str)
args_parser.add_argument('--wandb_proj', default='doge_universal', type=str)
args_parser.add_argument('--wandb_run', default=None, type=str)
args_parser.add_argument('--curriculum_path', default=None, type=str)
args_parser.add_argument('--cc_selection', action='store_true')
args_parser.add_argument('--cc_ns', default=10, type=int)
args_parser.add_argument('--cc_steps', default=1000, type=int)
args_parser.add_argument('--total_iterations', default=10000, type=int)

    
def main():
    args = args_parser.parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_proj # name your W&B project 
    config_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FullTrainingArguments))
    if args.config_json is not None:
        model_args, data_args, training_args = config_parser.parse_json_file(json_file=args.config_json)
    else:
        model_args, data_args, training_args = config_parser.parse_args_into_dataclasses()
    
    if args.wandb_run is None:
        wandb_run_name = training_args.run_name
    else:
        wandb_run_name = args.wandb_run
    
    training_args.local_rank = -1
    print("training local_rank: ", training_args.local_rank)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    train_ds, val_ds, domain_config, tokenizer, train_dataset_ls = get_train_eval_datasets(data_config=data_args,
                                                        verbose=True,
                                                        doremi=training_args.doremi,
                                                        ) 
    
    data_collator=get_data_collator(tokenizer, do_padding=data_args.do_padding, max_length=data_args.max_token_length)
    grad_acc_steps = training_args.gradient_accumulation_steps
    if training_args.doremi:
        # TODO: train reference model for doremi
        if training_args.ref_model is not None:
            logger.info("*** Load Reference Model (DoReMi) ***")
            ref_model, _ = get_model_from_config(model_args, doge=True, ref_model_path=training_args.ref_model)
        else:
            ref_model, _ = get_model_from_config(model_args, doge=True, ref_model_path=None)
            set_seed(training_args.seed)
            training_args.ddp_find_unused_parameters = False
            torch.cuda.empty_cache()
            # Initialize our Trainer
            ref_trainer = DoGETrainer(
                model=ref_model,
                args=training_args,
                domain_args=domain_config,
                train_dataset=train_ds if training_args.do_train else None,
                eval_dataset=val_ds if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                selected_modules_ls=None,
                wandb_run_name="ref_doremi_"+wandb_run_name,
                output_dir=os.path.join(training_args.output_dir, "ref_doremi_"+wandb_run_name),
                grad_acc=1,
            )
            if training_args.do_train:
                logger.info("*** Train Reference Model (DoReMi) ***")
                checkpoint = None
                ref_trainer.train(resume_from_checkpoint=None)
        ref_model.to("cuda")
    else:
        ref_model = None
    if args.cc_selection:
        cc_model, _ = get_model_from_config(model_args, doge=True)
        # Set seed before initializing model.
        set_seed(training_args.seed)
        # turn off find unused parameters
        training_args.ddp_find_unused_parameters = False

        torch.cuda.empty_cache()
        # Initialize our Trainer
        trainer = DoGETrainer(
            model=cc_model,
            args=training_args,
            domain_args=domain_config,
            train_dataset=train_ds if training_args.do_train else None,
            eval_dataset=val_ds if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            cc_selection=args.cc_selection,
            cc_ns=args.cc_ns,
            cc_steps=args.cc_steps,
            selected_modules_ls=None,
            wandb_run_name="cc_"+wandb_run_name,
            output_dir=os.path.join(training_args.output_dir, "cc_"+wandb_run_name),
            grad_acc=1,
        )
        if training_args.do_train:
            logger.info("*** Assessing Cancellation Effect ***")
            checkpoint = None
            trainer.train(resume_from_checkpoint=None)
            selected_modules_ls = trainer.selected_modules
            weight_dict = trainer.prev_w
            selected_params_num = 0
            for k in selected_modules_ls:
                selected_params_num += weight_dict[k].flatten().shape[0]
            print('Selected Modules: ')
            for m in selected_modules_ls:
                print('| ', m)
            print('Total parameters to compute W: ', selected_params_num, f'({selected_params_num*100/cc_model.num_parameters()}%)')
    else:
        selected_modules_ls = None
        selected_params_num = None
    
    ## Start Training ##
    # Detecting last checkpoint.
    doge_model, doge_config = get_model_from_config(model_args, doge=True)
    print("DoGE model parameters: ", doge_model.num_parameters())
    print("Num. GPU used: ", training_args.n_gpu)
    print("Gradient accumulate steps: ", training_args.gradient_accumulation_steps)
    last_checkpoint = None
    num_skip_examples = 0
    output_dir = os.path.join(training_args.output_dir, wandb_run_name)
    if os.path.isdir(output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            state = TrainerState.load_from_json(str(Path(last_checkpoint) / TRAINER_STATE_NAME))
            global_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
            num_skip_examples = state.global_step * global_batch_size
            logger.info(f"Skipping {num_skip_examples} examples")
    else:
        os.makedirs(output_dir, exist_ok=True)
    # Set seed before initializing model.
    set_seed(training_args.seed)
    # turn off find unused parameters
    training_args.ddp_find_unused_parameters = False

    torch.cuda.empty_cache()
    # Initialize our Trainer
    trainer = DoGETrainer(
        model=doge_model,
        args=training_args,
        domain_args=domain_config,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=val_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        selected_modules_ls=selected_modules_ls,
        selected_params_num=selected_params_num,
        cc_selection=None,
        cc_ns=None,
        cc_steps=None,
        wandb_run_name=wandb_run_name,
        output_dir=output_dir,
        total_iterations=args.total_iterations,
        grad_acc=grad_acc_steps,
        ref_model=ref_model,
        train_dataset_ls=train_dataset_ls,
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if training_args.eval_all_checkpoints:
            checkpoint_dir_list = trainer.get_all_checkpoints(training_args.output_dir)
        else:
            checkpoint_dir_list = [get_last_checkpoint(training_args.output_dir)]

        for checkpoint_dir in checkpoint_dir_list:
            trainer.load_checkpoint(checkpoint_dir)
            state = TrainerState.load_from_json(str(Path(checkpoint_dir) / TRAINER_STATE_NAME))
            trainer.state.global_step = state.global_step

            if not training_args.skip_perplexity_eval:
                metrics = trainer.evaluate()
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

            if training_args.downstream_datasets is not None:
                dataset_names = training_args.downstream_datasets.split(',')
                downstream_metrics = trainer.evaluate_fewshot(
                        dataset_names,
                        max_samples=data_args.max_downstream_samples,
                        num_shots=training_args.downstream_num_shots)
                trainer.log_metrics("eval", downstream_metrics)
                trainer.save_metrics("eval", downstream_metrics)

    print('DoGE launched! ❤️‍🔥❤️‍🔥')
        
if __name__ == "__main__":
    main()