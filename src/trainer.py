import math
import warnings
import json
import re
import os
import sys
import time
import shutil
import pickle
from pathlib import Path
import wandb
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from datasets import IterableDataset
import transformers
from transformers import Trainer
from transformers.utils import ExplicitEnum, is_torch_tpu_available
from transformers.optimization import get_scheduler
from transformers.utils import logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import (
        has_length,
        denumpify_detensorize,
        EvalLoopOutput,
        enable_full_determinism,
        set_seed,
        get_last_checkpoint,
        PREFIX_CHECKPOINT_DIR
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerState
from transformers.trainer import TRAINER_STATE_NAME
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import transformers
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers import (
    TrainingArguments, 
    MODEL_FOR_CAUSAL_LM_MAPPING,
    CONFIG_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from eval_datasets import get_eval_dataset
from dataloader import DataTrainingArguments, get_data_collator, get_train_eval_datasets, interleave_datasets
from models import CausalLMOutputWithDomainIDs, ModelArguments, get_model_from_config, GPT2DoGE
import logging
from accelerate import Accelerator, skip_first_batches


@dataclass
class FullTrainingArguments(TrainingArguments):
    lr_end: float = field(
            default=1e-4,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    reweight_domains: bool = field(
        default=True, metadata={"help": "Do reweighting."}
    )
    doremi: bool = field(
        default=False, metadata={"help": "DoReMi."}
    )
    ref_model: str = field(
        default=None, metadata={"help": "path to pretrained reference model (only used to run DoReMi)."}
    )
    lr_scheduler_name: str = field(
        default='linear_warmup_cosine', metadata={"help": "Custom LR scheduler name (linear_warmup_exponential, linear_warmup_cosine)"}
    )
    skip_perplexity_eval: bool = field(
        default=False, metadata={"help": "Don't evaluate perplexity."}
    )
    downstream_datasets: str = field(
            default='trivia_qa,web_questions,lambada,natural_questions,squad_v2', metadata={"help": "Comma-delimited list of dataset names from: {trivia_qa, web_questions, lambada, natural_questions, squad_v2}"}
    )
    eval_all_checkpoints: bool = field(
        default=False, metadata={"help": "Evaluate all the checkpoints at once."}
    )
    downstream_num_shots: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Number of in-context examples for downstream tasks. Defaults to 1"
            )
        },
    )
    reweight_eps: float = field(
        default=0.0,
        metadata={"help": "Smoothing factor."}
    )
    mu: float = field(
        default=0.05,
        metadata={"help": "Hyperparam for Bregman Divergence."}
    )
    dw_max: float = field(
        default=5.0,
        metadata={"help": "Score clip upper bound (*lr_t)."}
    )
    dw_min: float = field(
        default=0.00,
        metadata={"help": "Score clip lower bound (*lr_t)."}
    )
    compute_pertoken_losses: bool = field(
        default=True, metadata={"help": "Compute all domain losses at once."}
    )
    domain_update_per_iter: int = field(
        default=None, metadata={"help": "Number of domains update at each iteration."}
    )
    
    

def get_model_grad_flat(model, tgt_params_ls=None):
    ''' Get flattened gradient vectors for all layers. '''
    # make sure all grads are detached #
    full_grad_concat = None
    for p_name, p in model.named_parameters():
        if tgt_params_ls is not None and p_name not in tgt_params_ls:
            continue
        flat_grad = p.grad.detach().flatten()
        # add to full grad #
        if full_grad_concat is not None:
            full_grad_concat = torch.concat([full_grad_concat, flat_grad])
        else:
            full_grad_concat = flat_grad
    return full_grad_concat

def get_model_grad_flat_dict(model):
    ''' Get flattened gradient vectors for all param modules.
        Use to compute cancellation effect. '''
    # make sure all grads are detached #
    grad_vec_dict = {}
    for p_name, p in model.named_parameters():
        flat_grad = p.grad.detach().flatten()
        grad_vec_dict[p_name] = flat_grad
    return grad_vec_dict

def get_model_weights_flat_dict(model):
    ''' Get flattened weight vectors for all param modules.
        Use for compute cancellation effect. '''
    # make sure all weights are detached #
    weights_dict = {}
    for p_name, p in model.named_parameters():
        flat_w = p.detach().flatten()
        weights_dict[p_name] = flat_w
    return weights_dict

def get_model_grad_dict(model):
    # do not detach #
    full_grad_dict = {}
    for p_name, p in model.named_parameters():
        full_grad_dict[p_name] = p.grad
    return full_grad_dict

def add_model_grad_ls(model, domain_full_grad_dict, dw=None):
    if dw is None or type(domain_full_grad_dict)==dict:
      add_model_grad(model, domain_full_grad_dict)
    for p_name, p in model.named_parameters():
        for idx,v in enumerate(dw):
            if domain_full_grad_dict[idx] is not None:
                if p.grad is None:
                    p.grad = domain_full_grad_dict[idx][p_name]*v
                else:
                    p.grad += domain_full_grad_dict[idx][p_name]*v

def add_model_grad(model, domain_full_grad_dict):
    for p_name, p in model.named_parameters():
        if p.grad is None:
            p.grad = domain_full_grad_dict[p_name]
        else:
            p.grad += domain_full_grad_dict[p_name]
            

class LinearWarmupExponentialLR(LRScheduler):
    """
    Exponential LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # figure out decay rate to use to get within 1e-10 of lr_end at end of training
            gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_training_steps - self.num_warmup_steps))
                      for base_lr in self.base_lrs]
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for base_lr, gamma in zip(self.base_lrs, gammas)]


class LinearWarmupCosineLR(LRScheduler):
    """
    Cosine LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]


class ExtendedSchedulerType(ExplicitEnum):
    LINEAR_WARMUP_EXPONENTIAL = "linear_warmup_exponential"
    LINEAR_WARMUP_COSINE = "linear_warmup_cosine"


# extend scheduler function mapping
TYPE_TO_EXTENDED_SCHEDULER_FUNCTION = {
        ExtendedSchedulerType.LINEAR_WARMUP_EXPONENTIAL: LinearWarmupExponentialLR,
        ExtendedSchedulerType.LINEAR_WARMUP_COSINE: LinearWarmupCosineLR
}


def get_scheduler_extended(
    name,
    optimizer,
    num_warmup_steps=0,
    num_training_steps=0,
    lr_end=1e-4,
):

    try:
        name = ExtendedSchedulerType(name)
        schedule_func = TYPE_TO_EXTENDED_SCHEDULER_FUNCTION[name]
    except ValueError:
        return get_scheduler(name, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_end=lr_end)

### Trainer ###
import logging
logger = logging.getLogger(__name__)

class DoGETrainer(Trainer):
    def __init__(self, *args, domain_args, 
                 cc_selection=None, cc_ns=None, cc_steps=None, selected_modules_ls=None, selected_params_num=None, 
                 total_iterations=10000, wandb_run_name="test_test_test", output_dir=None,
                 grad_acc=None, ref_model=None, train_dataset_ls=None,
                 **kwargs,):
        ''' args to init the original Trainer
          model: Union[PreTrainedModel, nn.Module] = None,
          args: TrainingArguments = None,
          domain_args: DomainConfigArguments = None,
          data_collator: Optional[DataCollator] = None,
          train_dataset: Optional[Dataset] = None,
          eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
          tokenizer: Optional[PreTrainedTokenizerBase] = None,
          model_init: Optional[Callable[[], PreTrainedModel]] = None,
          compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
          callbacks: Optional[List[TrainerCallback]] = None,
          optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
          preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        '''
        super().__init__(*args, **kwargs)
        
        self.domain_config = domain_args
        self.train_dw = self.domain_config.train_dw
        self.val_dw = self.domain_config.val_dw
        self.train_ids = self.domain_config.train_ids
        self.tgt_ids = self.domain_config.tgt_ids

        self.idx2domain = self.domain_config.idx2domain
        self.domain2idx = self.domain_config.domain2idx
        self.domain_list = self.domain_config.domain_list
        
        tgt_domains = [self.idx2domain[d] for d in self.domain_config.tgt_ids.tolist()]
        self.eval_domain_list = self.domain_config.domain_list
        self.sampling_weights = self.train_dw

        self.reweight_eps = self.args.reweight_eps
        self.doge = self.args.reweight_domains 
        self.doremi = self.args.doremi 
        self.ref_model = ref_model
        self.mu = self.args.mu
        self.dw_min = self.args.dw_min
        self.dw_max = self.args.dw_max
        self.compute_pertoken_losses = self.args.compute_pertoken_losses
        self.cc_selection = cc_selection
        self.cc_ns = cc_ns
        self.cc_steps = cc_steps
        if grad_acc is not None:
            self.args.gradient_accumulation_steps = grad_acc
        
        if self.domain_config.curriculum_path is not None:
            with open(self.domain_config.curriculum_path, "rb") as trg:
                self.curriculum = pickle.load(trg)
            self.train_dataset_ls = train_dataset_ls
        else:
            self.curriculum = None
            self.train_dataset_ls = None
            
        if self.doremi:
            self.perdomain_scores = None
        if self.args.domain_update_per_iter is not None:
            self.domain_update_per_iter = self.args.domain_update_per_iter
            self.domain_update_counter = {i:0 for i in range(len(self.domain_list))}
        else:
            self.domain_update_per_iter = None
            self.domain_update_counter = {i:0 for i in range(len(self.domain_list))}
        if self.args.reweight_domains or self.args.doremi:
            self.train_dw = torch.ones(len(self.domain_list), dtype=torch.float)/len(self.train_ids)
            if len(self.domain_list)>len(self.train_ids):
                exclude_ids = torch.tensor([i for i in range(len(self.domain_list)) if i not in self.train_ids.numpy()])
                self.train_dw[exclude_ids] = 0.0
            if 'mix' not in tgt_domains:
                self.val_dw = torch.zeros(len(self.domain_list), dtype=torch.float)
                self.val_dw[self.tgt_ids] = 1/len(self.tgt_ids)
            else:
                self.val_dw = torch.ones(len(self.domain_list), dtype=torch.float)/len(self.domain_list)

        self.pertoken_losses_all = []
        self.token_masks = []
        self.domain_ids = []
        if selected_params_num is None:
            self.selected_params_num = self.model.num_parameters()
        else:
            self.selected_params_num = selected_params_num
        self.flat_grad_mat = torch.zeros((len(self.domain_list), self.selected_params_num), dtype=torch.float)
        self.grad_acc_step = 0
        
        self.perdomain_scores = torch.zeros(len(self.train_ids), dtype=torch.float)+1e-6
        self.avg_dw = torch.zeros(len(self.domain_list), dtype=torch.float)
        self.dw_update_steps = 0
        self.iter_domain_losses = torch.zeros(len(self.domain_list))
        self.args.run_name = wandb_run_name
        if output_dir is not None:
            self.args.output_dir = output_dir
        if self.cc_selection:
            self.prev_w = None
            self.cc_dict = {}
            self.selected_modules = None
            self.args.max_steps = self.cc_steps
        else:
            self.selected_modules = selected_modules_ls
            if self.selected_modules is not None:
                if "module" in self.selected_modules[0]:
                    self.selected_modules = [m[7:] for m in self.selected_modules]
                print('Selected Modules: ')
                for m in self.selected_modules:
                    print('| ', m)
            self.args.max_steps = total_iterations
        print(f'Training for {self.args.max_steps} Steps')
        if self.ref_model is not None:
            print("** Reference Model **")
            print(self.ref_model)
        print('Train-IDs 2 Domains:')
        for i in self.train_ids.tolist():
            print(f'{i}-{self.idx2domain[i]}')
        print('Target-IDs 2 Domains:')
        for i in self.tgt_ids.tolist():
            print(f'{i}-{self.idx2domain[i]}')
        print('==============================')
        print('Training DW: ')
        for idx,domain_name in self.idx2domain.items():
            print(f'{domain_name}: ', self.train_dw[idx])
        
        print('Eval DW: ')
        for idx,domain_name in self.idx2domain.items():
            print(f'{domain_name}: ', self.val_dw[idx])
        self.last_dw_save_path = os.path.join(self.args.output_dir, 'last_dw_config.pkl')
        self.avg_dw_save_path = os.path.join(self.args.output_dir, 'avg_dw_config.pkl')
        if os.path.exists(self.last_dw_save_path):
            with open(self.last_dw_save_path, 'rb') as trg:
                cur_domain_config_dict = pickle.load(trg)
                self.train_dw = cur_domain_config_dict['train_dw']
                self.dw_update_steps = cur_domain_config_dict['dw_update_steps']
            with open(self.avg_dw_save_path, 'rb') as trg:
                avg_domain_config_dict = pickle.load(trg)
                self.avg_dw = avg_domain_config_dict['train_dw'] * self.dw_update_steps
            print(f'Resume training from step {self.dw_update_steps}...')
            print('Last-step Domain Weights:', self.train_dw)
            print('Average Domain Weights:', self.avg_dw / self.dw_update_steps)
            
    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def write_weights(self, cur_weights, avg_weights):
        cur_domain_config_dict = {k:v for k,v in self.domain_config.__dict__.items()}
        avg_domain_config_dict = {k:v for k,v in self.domain_config.__dict__.items()}
        cur_domain_config_dict['train_dw'] = cur_weights
        cur_domain_config_dict['dw_update_steps'] = self.dw_update_steps
        avg_domain_config_dict['train_dw'] = avg_weights
        with open(self.last_dw_save_path, 'wb') as trg:
            pickle.dump(cur_domain_config_dict, trg)
        with open(self.avg_dw_save_path, 'wb') as trg:
            pickle.dump(avg_domain_config_dict, trg)
        
    def train_cancellation(self, model, inputs, ns=None, by_layer=False):
        model.train()
        if ns is None:
            ns = self.cc_ns
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False, return_pertoken_losses=False)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
        weight_dict = get_model_weights_flat_dict(model)
        grad_dict = get_model_grad_flat_dict(model)
        if self.prev_w is not None:
            for k, cur_w in weight_dict.items():
                assert cur_w is not None, f"Error getting weights on k={k}!"
                delta_w = cur_w - self.prev_w[k]
                grad_w = grad_dict[k]
                c = grad_w.norm()/delta_w.norm()
                if k not in self.cc_dict.keys():
                    self.cc_dict[k] = c
                else:
                    self.cc_dict[k] += c 
        self.prev_w = weight_dict    
        wandb_dict = {}
        for x,y in self.cc_dict.items():
            wandb_dict[f'cancellation/{x}'] = y
        wandb.log(wandb_dict)
        sorted_list = sorted(self.cc_dict.items(), key=lambda item: item[-1])
        if ns>0:
            self.selected_modules = [l[0] for l in sorted_list[:ns]]
        else:
            self.selected_modules = [l[0] for l in sorted_list[ns:]]
        return loss.detach()
        
    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_name is not None:
                lr_scheduler_name = self.args.lr_scheduler_name
            else:
                lr_scheduler_name = self.args.lr_scheduler_type
            self.lr_scheduler = get_scheduler_extended(
                lr_scheduler_name,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                lr_end=self.args.lr_end,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    
    def compute_loss(self, model, inputs, return_outputs=False, return_pertoken_losses=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs['return_pertoken_losses'] = return_pertoken_losses
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def update_domain_weights(self, pertoken_losses, token_masks, domain_ids):
        wandb_log_dict = {}
        domain_ids = domain_ids.detach()

        if self.doge:
            full_grad_dicts = []
            all_domain_losses = []
            for domain_id in range(len(self.domain_list)):
                self.model.zero_grad()
                domain_mask = (domain_ids == domain_id)
                if domain_mask.sum() > 0:
                    curr_domain_losses = pertoken_losses[token_masks*domain_mask.reshape(-1, 1)].mean()
                    all_domain_losses.append(curr_domain_losses)
                else:
                    all_domain_losses.append(None)
            
            for domain_id, curr_domain_losses in enumerate(all_domain_losses):
                if curr_domain_losses is None:
                    full_grad_dicts.append(None)
                else:
                    if self.use_apex:
                        with amp.scale_loss(curr_domain_losses, self.optimizer) as scaled_curr_domain_losses:
                            scaled_curr_domain_losses.backward()
                    else:
                        self.accelerator.backward(curr_domain_losses,retain_graph=True)
                    self.iter_domain_losses[domain_id] = curr_domain_losses.detach().cpu().item()
            
                    # get domain grad
                    if self.args.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    domain_flat_grad = get_model_grad_flat(self.model, tgt_params_ls=self.selected_modules)
                    domain_full_grad_dict = get_model_grad_dict(self.model)
                    self.flat_grad_mat[domain_id][:] = domain_flat_grad
                    full_grad_dicts.append(domain_full_grad_dict)
            train_mat = self.flat_grad_mat[self.train_ids][:]
            tgt_mat = self.flat_grad_mat[self.tgt_ids][:]
            scores_mat = train_mat @ tgt_mat.T
            
            lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 1e-4
            
            # TODO: Check dimension 
            if set(self.train_ids) == set(self.tgt_ids):
                scores = lr_t * (scores_mat.sum(dim=-1) - scores_mat.diag())
            else:
                scores = lr_t * scores_mat.sum(dim=-1)
            
            avg_norm = train_mat.norm(dim=-1).mean()
            scores = scores/(avg_norm+1e-6)
            scores = torch.clip(scores, min=lr_t*self.dw_min, max=lr_t*self.dw_max)
            
            dw_prev = self.train_dw
            log_dw_new = torch.log(dw_prev[self.train_ids]) + scores / self.mu
            dw_new = torch.nn.functional.softmax(log_dw_new, dim=-1)
            dw_new = (1-self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new) # default reweight_eps=0.0, no smoothing
            self.train_dw[self.train_ids] = dw_new
            self.avg_dw[self.train_ids] += dw_new
            self.dw_update_steps += 1
            add_model_grad_ls(self.model, [full_grad_dicts[i] for i in self.train_ids], dw=self.train_dw[self.train_ids])
            self.write_weights(cur_weights=self.train_dw, avg_weights=self.avg_dw/self.dw_update_steps)
        else:
            raise ValueError(f"Reweighting Scheme not supported")
        grad_norm = self.flat_grad_mat.norm(dim=-1)
        for domain_idx in range(len(self.domain_list)):
            domain_name = self.idx2domain[domain_idx]
            if domain_idx in self.train_ids:
                wandb_log_dict[f'score/{domain_name}'] = scores[domain_idx].item()
            elif domain_idx in self.tgt_ids:
                wandb_log_dict[f'score/{domain_name}'] = 0.0
            wandb_log_dict[f'grad_norm/{domain_name}'] = max(grad_norm[domain_idx].item(), self.args.max_grad_norm)
            wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
            wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            wandb_log_dict[f'loss/{domain_name}'] = self.iter_domain_losses[domain_idx]
        wandb_log_dict['lr'] = lr_t
        
        wandb.log(wandb_log_dict, commit=False)
    
    def filter_inputs(self, inputs, domain_id):
        selected_ids = inputs['domain_ids']==domain_id
        if selected_ids.sum()==0:
            return None
        
        new_inputs = {}
        for k,v in inputs.items():
            new_inputs[k] = v[selected_ids.flatten()]
        return new_inputs
        
    def train_step_distributed(self, model, inputs):
        self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        loss_all = torch.tensor(0.0)
        effective_domains = 0
        sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
        for i,c in sample_counter.items():
            if i in self.domain_update_counter.keys():
                self.domain_update_counter[i] += c
        
        self.grad_acc_step += 1
        for domain_id in range(len(self.domain_list)):
            new_inputs = self.filter_inputs(inputs, domain_id)
            if new_inputs is None:
                continue
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, new_inputs, return_outputs=True, return_pertoken_losses=False)
            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_losses = [
                            torch.zeros_like(loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(loss, gathered_losses, dst=0)
                    gathered_losses = torch.cat(gathered_losses, dim=0)
                    self.domain_losses_distributed[domain_id] += gathered_losses
                    loss_all += gathered_losses.detach().cpu()
                else:
                    dist.gather(loss, dst=0)
            else:
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                self.domain_losses_distributed[domain_id] = loss+self.domain_losses_distributed[domain_id]
                loss_all += loss.detach().cpu()
            effective_domains += 1
        if self.grad_acc_step == self.args.gradient_accumulation_steps:
            # TODO: update domain weights (DOGE)
            if self.args.gradient_accumulation_steps > 1:
                self.domain_losses_distributed = [l / self.args.gradient_accumulation_steps for l in self.domain_losses_distributed]
            self.update_domain_weights_distributed(self.domain_losses_distributed)

            self.grad_acc_step = 0 
            self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        return loss_all / (self.args.gradient_accumulation_steps*effective_domains)
    
    def update_domain_weights_distributed(self, domain_losses_distributed):
        wandb_log_dict = {}
        full_grad_dicts = []
        for domain_id in range(len(self.domain_list)):
            self.model.zero_grad()
            curr_domain_losses = domain_losses_distributed[domain_id]
            if curr_domain_losses > 0.0:
                if self.use_apex:
                    with amp.scale_loss(curr_domain_losses, self.optimizer) as scaled_curr_domain_losses:
                        scaled_curr_domain_losses.backward()
                else:
                    self.accelerator.backward(curr_domain_losses)
                self.iter_domain_losses[domain_id] = curr_domain_losses.detach().cpu().item()
                # get domain grad
                if self.args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                domain_flat_grad = get_model_grad_flat(self.model, tgt_params_ls=self.selected_modules)
                
                self.flat_grad_mat[domain_id][:] = domain_flat_grad
                if domain_id not in self.train_ids:
                    full_grad_dicts.append(None)
                else:
                    domain_full_grad_dict = get_model_grad_dict(self.model)
                    full_grad_dicts.append(domain_full_grad_dict)
            else:
                full_grad_dicts.append(None)
            self.model.zero_grad()
        train_mat = self.flat_grad_mat[self.train_ids][:]
        tgt_mat = self.flat_grad_mat[self.tgt_ids][:]
        scores_mat = train_mat @ tgt_mat.T
        
        lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 1e-4
        
        # TODO: Check dimension 
        if set(self.train_ids) == set(self.tgt_ids):
            scores = lr_t * (scores_mat.sum(dim=-1) - scores_mat.diag())
        else:
            scores = lr_t * scores_mat.sum(dim=-1)
        
        avg_norm = train_mat.norm(dim=-1).mean()
        scores = scores/(avg_norm+1e-6)
        scores = torch.clip(scores, min=lr_t*self.dw_min, max=lr_t*self.dw_max)
        
        dw_prev = self.train_dw
        log_dw_new = torch.log(dw_prev[self.train_ids]) + scores / self.mu
        dw_new = torch.nn.functional.softmax(log_dw_new, dim=-1)
        dw_new = (1-self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new) # default reweight_eps=0.0, no smoothing
        self.train_dw[self.train_ids] = dw_new
        self.avg_dw[self.train_ids] += dw_new
        self.dw_update_steps += 1
        add_model_grad_ls(self.model, [full_grad_dicts[i] for i in self.train_ids], dw=self.train_dw[self.train_ids])
        self.write_weights(cur_weights=self.train_dw, avg_weights=self.avg_dw/self.dw_update_steps)
        
        grad_norm = self.flat_grad_mat.norm(dim=-1)
        for domain_idx in range(len(self.domain_list)):
            domain_name = self.idx2domain[domain_idx]
            if domain_idx in self.train_ids:
                score_idx = self.train_ids.tolist().index(domain_idx)
                wandb_log_dict[f'score/{domain_name}'] = scores[score_idx].item()
            elif domain_idx in self.tgt_ids:
                wandb_log_dict[f'score/{domain_name}'] = 0.0
            wandb_log_dict[f'grad_norm/{domain_name}'] = grad_norm[domain_idx].item()
            wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
            wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            wandb_log_dict[f'loss/{domain_name}'] = self.iter_domain_losses[domain_idx]
            if domain_idx in self.domain_update_counter.keys():
                wandb_log_dict[f'sample_count/{domain_name}'] = self.domain_update_counter[domain_idx]    
        wandb_log_dict['lr'] = lr_t
        
        wandb.log(wandb_log_dict, commit=False)
    
    def train_step_doremi(self, model, inputs):
        assert self.doremi, "Only run this function for doremi!"
        self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        self.ref_domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        
        loss_all = torch.tensor(0.0)
        ref_loss_all = torch.tensor(0.0)
        effective_domains = 0
        sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
        for i,c in sample_counter.items():
            if i in self.domain_update_counter.keys():
                self.domain_update_counter[i] += c
        self.grad_acc_step += 1
        for domain_id in range(len(self.domain_list)):
            new_inputs = self.filter_inputs(inputs, domain_id)
            if new_inputs is None:
                continue
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, new_inputs, return_outputs=True, return_pertoken_losses=False)
                ref_loss, ref_outputs = self.compute_loss(self.ref_model, new_inputs, return_outputs=True, return_pertoken_losses=False)
            
            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_losses = [
                            torch.zeros_like(loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(loss, gathered_losses, dst=0)
                    gathered_losses = torch.cat(gathered_losses, dim=0)
                    self.domain_losses_distributed[domain_id] += gathered_losses
                    loss_all += gathered_losses.detach().cpu()
                    
                    ref_gathered_losses = [
                            torch.zeros_like(ref_loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(ref_loss, ref_gathered_losses, dst=0)
                    ref_gathered_losses = torch.cat(ref_gathered_losses, dim=0)
                    self.ref_domain_losses_distributed[domain_id] += ref_gathered_losses
                    ref_loss_all += ref_gathered_losses.detach().cpu()
                else:
                    dist.gather(loss, dst=0)
                    dist.gather(ref_loss, dst=0)
            else:
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                    ref_loss = ref_loss.mean()
                self.domain_losses_distributed[domain_id] = loss+self.domain_losses_distributed[domain_id]
                self.ref_domain_losses_distributed[domain_id] = ref_loss+self.ref_domain_losses_distributed[domain_id]
                ref_loss_all += ref_loss.detach().cpu()
                loss_all += loss.detach().cpu()
            effective_domains += 1
        if self.grad_acc_step == self.args.gradient_accumulation_steps:
            # TODO: update domain weights (DOGE)
            if self.args.gradient_accumulation_steps > 1:
                self.domain_losses_distributed = [l / self.args.gradient_accumulation_steps for l in self.domain_losses_distributed]
                self.ref_domain_losses_distributed = [l / self.args.gradient_accumulation_steps for l in self.ref_domain_losses_distributed]
            self.update_domain_weights_doremi(self.domain_losses_distributed, self.ref_domain_losses_distributed)

            self.grad_acc_step = 0 
            self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
            self.ref_domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        return loss_all / (self.args.gradient_accumulation_steps*effective_domains)
    
    def update_domain_weights_doremi(self, domain_losses_distributed, ref_domain_losses_distributed):
        assert self.doremi, "Only run this function for doremi!"
        excess_losses = torch.tensor([domain_losses_distributed[i]-ref_domain_losses_distributed[i] for i in range(len(domain_losses_distributed))], dtype=torch.float)
        for i in range(len(excess_losses)):
            if (domain_losses_distributed[i]>0.0 or self.perdomain_scores is None):
                continue
            excess_losses[i] = self.perdomain_scores[i]
        wandb_log_dict = {}
        
        excess_losses = torch.clip(excess_losses, min=0.0)        
        self.perdomain_scores = excess_losses.detach().cpu().tolist()
        lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else self.args.learning_rate
        log_new_train_dw = torch.log(self.train_dw) + 0.1 * excess_losses
        log_new_train_dw = log_new_train_dw - torch.logsumexp(log_new_train_dw, dim=0) # softmax normalization
        # smoothing
        dw_new = (1-self.reweight_eps) * torch.exp(log_new_train_dw) + self.reweight_eps / len(log_new_train_dw)
        
        self.train_dw[self.train_ids] = dw_new
        self.avg_dw[self.train_ids] += dw_new
        self.dw_update_steps += 1
        self.write_weights(cur_weights=self.train_dw, avg_weights=self.avg_dw/self.dw_update_steps)
    
        for domain_idx in range(len(self.domain_list)):
            domain_name = self.idx2domain[domain_idx]
            
            if domain_idx in self.train_ids:
                score_idx = self.train_ids.tolist().index(domain_idx)
                curr_domain_losses = domain_losses_distributed[score_idx] * dw_new[score_idx]
                if curr_domain_losses > 0.0:
                    if self.use_apex:
                        with amp.scale_loss(curr_domain_losses, self.optimizer) as scaled_curr_domain_losses:
                            scaled_curr_domain_losses.backward()
                    else:
                        self.accelerator.backward(curr_domain_losses)
                    if self.args.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
                wandb_log_dict[f'score/{domain_name}'] = excess_losses[score_idx].item()
                wandb_log_dict[f'loss/{domain_name}'] = domain_losses_distributed[score_idx]
            wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
            wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            if domain_idx in self.domain_update_counter.keys():
                wandb_log_dict[f'sample_count/{domain_name}'] = self.domain_update_counter[domain_idx]    
        
        lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else self.args.learning_rate
        wandb_log_dict['lr'] = lr_t
        
        wandb.log(wandb_log_dict, commit=False)
    
    def train_step_selected(self, model, inputs):
        assert self.domain_update_per_iter is not None, "You are updating all domains per iter, call `train_step`"
        self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        loss_all = torch.tensor(0.0)
        effective_domains = self.domain_update_per_iter
        self.grad_acc_step += 1
        sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
        selected_domains = [i[0] for i in sorted(sample_counter.items(), key=lambda item: item[1])[:effective_domains]]
        for i in selected_domains:
            if i in self.domain_update_counter.keys():
                self.domain_update_counter[i] += sample_counter[i]
        selected_domains = list(set(selected_domains + self.tgt_ids.tolist()))
        
        for domain_id in selected_domains:
            new_inputs = self.filter_inputs(inputs, domain_id)
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, new_inputs, return_outputs=True, return_pertoken_losses=False)
            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_losses = [
                            torch.zeros_like(loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(loss, gathered_losses, dst=0)
                    gathered_losses = torch.cat(gathered_losses, dim=0)
                    self.domain_losses_distributed[domain_id] += gathered_losses
                else:
                    dist.gather(loss, dst=0)
            else:
                self.domain_losses_distributed[domain_id] = loss+self.domain_losses_distributed[domain_id]
            loss_all += loss.detach().cpu()
        if self.grad_acc_step == self.args.gradient_accumulation_steps:
            # TODO: update domain weights (DOGE)
            if self.args.gradient_accumulation_steps > 1:
                self.domain_losses_distributed = self.domain_losses_distributed / self.args.gradient_accumulation_steps
            self.update_domain_weights_selected(domain_losses_selected=self.domain_losses_distributed,
                                                selected_domains=selected_domains)

            self.grad_acc_step = 0 
            self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        return loss_all / (self.args.gradient_accumulation_steps*effective_domains)
    
    def update_domain_weights_selected(self, domain_losses_selected, selected_domains):
        wandb_log_dict = {}
        full_grad_dicts = []
        for domain_id in range(len(self.domain_list)):
            self.model.zero_grad()
            curr_domain_losses = domain_losses_selected[domain_id]
            if (domain_id in selected_domains) and curr_domain_losses > 0.0:
                if self.use_apex:
                    with amp.scale_loss(curr_domain_losses, self.optimizer) as scaled_curr_domain_losses:
                        scaled_curr_domain_losses.backward()
                else:
                    self.accelerator.backward(curr_domain_losses)
                self.iter_domain_losses[domain_id] = curr_domain_losses.detach().cpu().item()
                # get domain grad
                domain_flat_grad = get_model_grad_flat(self.model, tgt_params_ls=None)
                domain_full_grad_dict = get_model_grad_dict(self.model)
                self.flat_grad_mat[domain_id][:] = domain_flat_grad
                full_grad_dicts.append(domain_full_grad_dict)
            else:
                full_grad_dicts.append(None)
        selected_train_domains = torch.tensor([i for i in selected_domains if i in self.train_ids.tolist()])
        train_mat = self.flat_grad_mat[selected_train_domains][:]
        tgt_mat = self.flat_grad_mat[self.tgt_ids][:]
        scores_mat = train_mat @ tgt_mat.T
        
        lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 1e-4
        
        # TODO: Check dimension 
        if set(self.train_ids) == set(self.tgt_ids):
            scores = lr_t * (scores_mat.sum(dim=-1) - scores_mat.diag())
        else:
            scores = lr_t * scores_mat.sum(dim=-1)
        
        avg_norm = train_mat.norm(dim=-1).mean()
        scores = scores/(avg_norm+1e-6)
        scores = torch.clip(scores, min=lr_t*self.dw_min, max=lr_t*self.dw_max)
        self.perdomain_scores[selected_train_domains] = scores
        
        dw_prev = self.train_dw
        log_dw_new = torch.log(dw_prev[self.train_ids]) + self.perdomain_scores / self.mu
        dw_new = torch.nn.functional.softmax(log_dw_new, dim=-1)
        dw_new = (1-self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new) # default reweight_eps=0.0, no smoothing
        self.train_dw[self.train_ids] = dw_new
        self.avg_dw[self.train_ids] += dw_new
        self.dw_update_steps += 1
        add_model_grad_ls(self.model, [full_grad_dicts[i] for i in self.train_ids], dw=self.train_dw[self.train_ids])
        self.write_weights(cur_weights=self.train_dw, avg_weights=self.avg_dw/self.dw_update_steps)
        
        grad_norm = self.flat_grad_mat.norm(dim=-1)
        for domain_idx in range(len(self.domain_list)):
            domain_name = self.idx2domain[domain_idx]
            if domain_idx in self.train_ids:
                wandb_log_dict[f'score/{domain_name}'] = self.perdomain_scores[domain_idx].item()
            elif domain_idx in self.tgt_ids:
                wandb_log_dict[f'score/{domain_name}'] = 0.0
            wandb_log_dict[f'grad_norm/{domain_name}'] = grad_norm[domain_idx].item()
            wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
            wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            wandb_log_dict[f'loss/{domain_name}'] = self.iter_domain_losses[domain_idx]
            if domain_idx in self.domain_update_counter.keys():
                wandb_log_dict[f'sample_count/{domain_name}'] = self.domain_update_counter[domain_idx]
        wandb_log_dict['lr'] = lr_t
        
        wandb.log(wandb_log_dict, commit=False)
    
    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # if self.curriculum is not None:
        #     if (self.state.global_step>0) and self.state.global_step in self.curriculum.keys():
        #         self.set_sample_dw()
        inputs = self._prepare_inputs(inputs)
        # sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
        # print(sample_counter)
        if self.cc_selection:
            return self.train_cancellation(model, inputs)
        if self.doremi:
            return self.train_step_doremi(model, inputs)
        elif self.doge:
            if not self.compute_pertoken_losses:
                if self.domain_update_per_iter is not None:
                    return self.train_step_selected(model, inputs)
                return self.train_step_distributed(model, inputs)
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True, return_pertoken_losses=True)
                pertoken_loss = outputs.pertoken_loss
                token_mask = outputs.token_mask

            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_pertoken_losses = [
                            torch.zeros_like(pertoken_loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(pertoken_loss, gathered_pertoken_losses, dst=0)
                    gathered_pertoken_losses = torch.cat(gathered_pertoken_losses, dim=0)

                    gathered_token_mask = [
                            torch.zeros_like(token_mask) for _ in range(self.args.world_size)
                            ]
                    dist.gather(token_mask, gathered_token_mask, dst=0)
                    gathered_token_mask = torch.cat(gathered_token_mask, dim=0)

                    gathered_domain_id = [
                            torch.zeros_like(inputs['domain_ids']) for _ in range(self.args.world_size)
                            ]
                    dist.gather(inputs['domain_ids'], gathered_domain_id, dst=0)
                    gathered_domain_id = torch.cat(gathered_domain_id, dim=0)

                    self.pertoken_losses_all.append(gathered_pertoken_losses)
                    self.token_masks.append(gathered_token_mask.detach())
                    self.domain_ids.append(gathered_domain_id.detach())

                    if len(self.pertoken_losses_all) == self.args.gradient_accumulation_steps:
                        pertoken_losses_all = torch.cat(self.pertoken_losses_all, dim=0)
                        token_masks = torch.cat(self.token_masks, dim=0).bool()
                        domain_ids = torch.cat(self.domain_ids, dim=0)

                        # TODO: update domain weights (DOGE)
                        if self.args.gradient_accumulation_steps > 1:
                            pertoken_losses_all = pertoken_losses_all / self.args.gradient_accumulation_steps
                        self.update_domain_weights(pertoken_losses_all, token_masks, domain_ids)

                        self.pertoken_losses_all = []
                        self.token_masks = []
                        self.domain_ids = []
                else:
                    dist.gather(pertoken_loss, dst=0)
                    dist.gather(token_mask, dst=0)
                    dist.gather(inputs['domain_ids'], dst=0)
            else:
                self.pertoken_losses_all.append(pertoken_loss)
                self.token_masks.append(token_mask.detach())
                self.domain_ids.append(inputs['domain_ids'].detach())

                if len(self.pertoken_losses_all) == self.args.gradient_accumulation_steps:
                    pertoken_losses_all = torch.cat(self.pertoken_losses_all, dim=0)
                    token_masks = torch.cat(self.token_masks, dim=0).bool()
                    domain_ids = torch.cat(self.domain_ids, dim=0)

                    # TODO: update domain weights (DOGE)
                    if self.args.gradient_accumulation_steps > 1:
                        pertoken_losses_all = pertoken_losses_all / self.args.gradient_accumulation_steps
                    self.update_domain_weights(pertoken_losses_all, token_masks, domain_ids)

                    self.pertoken_losses_all = []
                    self.token_masks = []
                    self.domain_ids = []
            return loss.detach() / self.args.gradient_accumulation_steps
        else:
            wandb_log_dict = {}
            sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
            for i,c in sample_counter.items():
                if i in self.domain_update_counter.keys():
                    self.domain_update_counter[i] += c
            for domain_idx in self.domain_update_counter.keys():
                domain_name = self.idx2domain[domain_idx]
                wandb_log_dict[f'sample_count/{domain_name}'] = self.domain_update_counter[domain_idx]    
            wandb.log(wandb_log_dict, commit=False)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False, return_pertoken_losses=False)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            return loss.detach()
    #################################################################
    
    def load_checkpoint(self, resume_from_checkpoint=None):
        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(None)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint is None:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)

        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None and not is_sagemaker_mp_enabled() and self.args.deepspeed is None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, self.args.device)
            self.model_wrapped = self.model

    def get_all_checkpoints(self, folder):
        _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
        folder = Path(folder)
        checkpoints = [
            path
            for path in folder.iterdir()
            if _re_checkpoint.search(path.name) is not None and path.is_dir()
        ]
        checkpoints = list(sorted(checkpoints, key=lambda x: int(x.name.split('-')[1])))
        checkpoints = [str(path) for path in checkpoints]
        return checkpoints

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Computes per-domain log-perplexity, uniformly averaged log-perplexity, and worst-case log-perplexity
        """
        args = self.args

        if prediction_loss_only:
            # hack - don't do prediction loss only
            prediction_loss_only = None

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        if args.past_index >= 0:
            self._past = None

        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        losses = torch.zeros(len(self.eval_domain_list)).cuda()
        tokencounts = torch.zeros(len(self.eval_domain_list)).cuda()
        examplecounts = torch.zeros(len(self.eval_domain_list)).cuda()
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            domain_ids = inputs["domain_ids"].to(loss.device)

            if isinstance(logits, tuple):
                logits = logits[0]

            # compute losses per domain
            for domain_idx, domain_name in enumerate(self.eval_domain_list):
                domain_mask = (domain_ids == domain_idx).flatten()
                examplecounts[domain_idx] = examplecounts[domain_idx] + domain_mask.sum()

                if domain_mask.sum() > 0:
                    domain_labels = labels[domain_mask]
                    domain_preds = logits[domain_mask]
                    domain_labels = domain_labels[:, 1:].contiguous().view(-1)
                    domain_preds = domain_preds[:, :-1, :].contiguous().view(-1, domain_preds.size(-1))
                    losses[domain_idx] = losses[domain_idx] + loss_fn(domain_preds, domain_labels)
                    tokencounts[domain_idx] = tokencounts[domain_idx] + (domain_labels != -100).sum()

        if self.args.world_size>1:
            torch.distributed.all_reduce(losses)
            torch.distributed.all_reduce(tokencounts)
            torch.distributed.all_reduce(examplecounts)

        # losses/preds/labels on CPU (final containers)
        per_domain_losses = {domain_name: losses[domain_idx].item()
                             for domain_idx, domain_name in enumerate(self.eval_domain_list) if tokencounts[domain_idx] > 0}
        per_domain_tokencounts = {domain_name: tokencounts[domain_idx].item()
                                  for domain_idx, domain_name in enumerate(self.eval_domain_list) if tokencounts[domain_idx] > 0}
        per_domain_examplecounts = {domain_name: examplecounts[domain_idx].item()
                                    for domain_idx, domain_name in enumerate(self.eval_domain_list) if tokencounts[domain_idx] > 0}

        # normalize
        per_domain_losses = {domain_name: per_domain_losses[domain_name] / per_domain_tokencounts[domain_name]
                             for domain_name in per_domain_losses.keys()}

        metrics = {f"{domain_name}:log_perplexity": per_domain_losses[domain_name]
                   for domain_name in per_domain_losses.keys()}
        metrics["uniform_avg_log_ppl"] = np.mean(list(per_domain_losses.values()))
        metrics["worst_case_log_ppl"] = np.amax(list(per_domain_losses.values()))

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=sum(list(per_domain_examplecounts.values())))

    def evaluate_fewshot(self, dataset_names, ignore_keys=None, metric_key_prefix="eval", num_shots=1, max_samples=None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        max_token_length = self.tokenizer.model_max_length
        # prepare tokenizer
        tokenizer = self.tokenizer
        tokenizer_padding_side = tokenizer.padding_side
        tokenizer_truncation_side = tokenizer.truncation_side
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'

        all_metrics = {}

        for dataset_name in dataset_names:
            logger.info(f"Evaluating {dataset_name}...")

            # we pass in num_shots because some datasets are only 0 shot
            data_dict = get_eval_dataset(dataset_name, num_shots=num_shots, seed=self.args.seed)

            dataset_train = data_dict['dataset_train']
            dataset_val = data_dict['dataset_val']
            top_k = data_dict['top_k']
            top_p = data_dict['top_p']
            temperature = data_dict['temperature']
            prompt_transform = data_dict['prompt_transform']
            eval_func = data_dict['eval_func']
            pred_postprocess_func = data_dict['pred_postprocess_func']
            num_shots = data_dict['num_shots']
            max_new_tokens = data_dict['max_new_tokens']
            shuffle_train = data_dict['shuffle_train']

            # use training set as few-shot examples, shuffle first
            if dataset_train is not None and shuffle_train:
                dataset_train = dataset_train.shuffle(seed=self.args.seed)

            # select first num examples
            if max_samples is not None:
                dataset_val = dataset_val.select(range(max_samples))

            # shard the dataset
            if dataset_train is not None:
                dataset_train = dataset_train.shard(num_shards=self.args.world_size, index=self.args.process_index)
            dataset_val = dataset_val.shard(num_shards=self.args.world_size, index=self.args.process_index)

            def few_shot_generator(ds=None):
                while True:
                    curr_exs = []
                    if num_shots == 0:
                        yield curr_exs
                        continue

                    for ex in ds:
                        curr_exs.append(ex)

                        if len(curr_exs) == num_shots:
                            yield curr_exs
                            curr_exs = []

            fewshot_train_dataset = IterableDataset.from_generator(
                    few_shot_generator, gen_kwargs={'ds': dataset_train})

            def prompt_generator(fewshot_train_ds, val_ds):
                for ex, context_exs in zip(val_ds, fewshot_train_ds):
                    ex_dict = prompt_transform(ex, context_exs)
                    yield ex_dict

            def data_collator(batch):
                # self.tokenizer is the HF tokenizer
                # tokenizer is either HF tokenizer or SPM tokenizer
                collated_batch = {k: [f[k] for f in batch] for k in batch[0].keys()}
                # will do left truncation
                tokenized = tokenizer(collated_batch['prompt'], padding=False, truncation=True)

                collated_batch['input_ids'] = torch.tensor(tokenized['input_ids'])[:, -(max_token_length-max_new_tokens):]
                collated_batch['attention_mask'] = torch.tensor(tokenized['attention_mask'])[:, -(max_token_length-max_new_tokens):]
                return collated_batch

            fewshot_val_dataset = IterableDataset.from_generator(
                    prompt_generator, gen_kwargs={'fewshot_train_ds': fewshot_train_dataset, 'val_ds': dataset_val})

            dataloader = DataLoader(
                    fewshot_val_dataset,
                    batch_size=1,  # batch size 1 avoids left padding
                    collate_fn=data_collator,
                    num_workers=1,
                    pin_memory=self.args.dataloader_pin_memory)

            # prepare model
            args = self.args
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)

            # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
            # while ``train`` is running, cast it to the right dtype first and then put on device
            if not self.is_in_train:
                if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
                elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)
            model.eval()

            # TODO put this somewhere else?
            try:
                model.config.pad_token_id = model.config.eos_token_id
            except:
                model.module.config.pad_token_id = model.module.config.eos_token_id

            num_correct = torch.tensor(0.0).cuda()
            num_examples = torch.tensor(0.0).cuda()
            # fewshot eval loop
            for step, inputs in tqdm(enumerate(dataloader)):
                num_examples += len(inputs['input_ids'])
                with torch.no_grad():
                    try:
                        with self.compute_loss_context_manager():
                            gen_tokens = model.generate(
                                    input_ids=inputs['input_ids'].cuda(),
                                    max_length=inputs['input_ids'].shape[1]+max_new_tokens,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature)
                    except:
                        with self.compute_loss_context_manager():
                            gen_tokens = model.module.generate(
                                    input_ids=inputs['input_ids'].cuda(),
                                    max_length=inputs['input_ids'].shape[1]+max_new_tokens,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature)

                gen_text = tokenizer.batch_decode(gen_tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                for prompt, pred, answer in zip(inputs['prompt'], gen_text, inputs['answer']):
                    pred = pred_postprocess_func(pred)
                    if eval_func(answer, pred, prompt,
                                 model=model,
                                 tokenizer=tokenizer,
                                 inputs=inputs,
                                 trainer=self):
                        num_correct += 1
                        print(f"\033[0;32m CORRECT \033[0m: {prompt}\033[0;32m{pred}\033[0m |  Answer: {answer}\n")
                    else:
                        print(f"\033[91m INCORRECT \033[0m: {prompt}\033[91m{pred}\033[0m |  Answer: {answer}\n")
            if self.args.world_size>1:
                torch.distributed.all_reduce(num_correct)
                torch.distributed.all_reduce(num_examples)
            accuracy = 100 * (num_correct / num_examples)

            metrics = {'accuracy': accuracy, 'num_correct': num_correct, 'num_examples': num_examples}

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_{num_shots}-shot:{dataset_name}"):
                    metrics[f"{metric_key_prefix}_{num_shots}-shot:{dataset_name}:{key}"] = metrics.pop(key)

            all_metrics.update(metrics)

        # comput average metrics across datasets
        avg_metrics = defaultdict(list)
        for key in all_metrics:
            if key.endswith('accuracy'):
                avg_metrics['accuracy'].append(all_metrics[key])
            if key.endswith('num_correct'):
                avg_metrics['num_correct'].append(all_metrics[key])
            if key.endswith('num_examples'):
                avg_metrics['num_examples'].append(all_metrics[key])

        avg_metrics = {key: np.mean(val_list) for key, val_list in avg_metrics.items()}

        for key in avg_metrics.keys():
            all_metrics[f"{metric_key_prefix}_{num_shots}-shot:avg:{key}"] = avg_metrics[key]

        # gather and compute metrics
        output = EvalLoopOutput(predictions=None, label_ids=None, metrics=all_metrics, num_samples=None)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # restore tokenizer settings
        tokenizer.padding_side = tokenizer_padding_side
        tokenizer.truncation_side = tokenizer_truncation_side

        return output.metrics
    
    def _inner_training_loop_curriculum(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                from packaging import version
                from accelerate import __version__ as accelerate_version
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    from accelerate.data_loader import SeedableRandomSampler
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            while self.state.global_step < steps_in_epoch:
                epoch_iterator = train_dataloader
                for step, inputs in enumerate(epoch_iterator):
                    total_batched_samples += 1

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)    
                        
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
                    
                    if (self.curriculum is not None) and (self.state.global_step>0) and (self.state.global_step in self.curriculum.keys()):
                        new_sample_dw = torch.tensor(self.curriculum[self.state.global_step], dtype=torch.float)
                        train_dataloader.dataset._ex_iterable.probabilities_handle = new_sample_dw
                        train_dataloader.dataset._ex_iterable.probabilities = new_sample_dw
                        print(f"Current Training Domain Weights (step={self.state.global_step}):", new_sample_dw)

                        break
                    
                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self.curriculum is not None:
            return self._inner_training_loop_curriculum(batch_size=batch_size, args=args, resume_from_checkpoint=resume_from_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval)
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                from packaging import version
                from accelerate import __version__ as accelerate_version
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    from accelerate.data_loader import SeedableRandomSampler
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if  not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
            # while self.state.global_step < steps_in_epoch:
                # inputs = next(epoch_iterator.__iter__())
                # step += 1
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
    
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
