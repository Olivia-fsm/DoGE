## Data Arguments ##
from dataclasses import dataclass, field
import pickle
# from typing import Optional
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import TrainingArguments, MODEL_FOR_CAUSAL_LM_MAPPING
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import transformers
from data.utils import get_dataset
import random
import numpy as np
from datasets import Dataset, IterableDataset, load_from_disk
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from pathlib import Path
from collections import Counter
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from torch.utils.data import WeightedRandomSampler
from cfg_tokenizer import CFGTokenizer
# transformers.utils.move_cache('/mloraw1/sfan/huggingface_cache')

RANDOM_BATCH_SIZE = 512
DEFAULT_SEED=111


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: str = field(
        default='.', metadata={"help": "Path to the dataset directory."}
    )
    dataset: str = field(
        default='redpajama-all', metadata={"help": "Name of the dataset."}
    )
    curriculum_path: str = field(
        default=None, metadata={"help": "Path to stage-wise curriculum domain weights (.pkl)."}
    )
    train_domains: str = field(
        default='arxiv,book,cc,c4,github,wikipedia', metadata={"help": "domain names for training."}
    )
    tgt_domains: str = field(
        default='stackexchange', metadata={"help": "target domain name(s) for generalization."}
    )
    train_dw: str = field(
        default=None, metadata={"help": "training domain weights."}
    )
    val_dw: str = field(
        default=None, metadata={"help": "validation domain weights."}
    )
    eval_dataset_dir: str = field(
        default=None, metadata={"help": "Path to the eval dataset directory. Defaults to dataset_dir"}
    )
    eval_dataset_name: str = field(
        default=None, metadata={"help": "Name of the eval dataset. Defaults to dataset_name."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_downstream_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For quicker downstream evaluation, limit the number of examples if set."
            )
        },
    )
    max_token_length: int = field(
        default=512,
        metadata={
            "help": (
                "Input sequence length after tokenization. "
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    do_padding: bool = field(
        default=False, metadata={"help": "Pad the inputs."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Shuffle the training data on the fly"}
    )
    keep_in_memory: bool = field(
        default=False, metadata={"help": "keep data in memory"}
    )
    


@dataclass
class DomainConfigArguments:
    """
    Domain config settings. """

    domain_list: list = field(
        default_factory=list,
        # default=['arxiv', 'book', 'c4', 'cc', 'github', 'stackexchange', 'wikipedia'], 
        metadata={"help": "List of domain names."}
    )
    train_dw: torch.Tensor = field(
        default=None, 
        metadata={"help": "Training domain weights."}
    )
    val_dw: torch.Tensor = field(
        default=None, 
        metadata={"help": "Validation domain weights."}
    )
    idx2domain: dict = field(
        default_factory=dict,
        metadata={"help": "index mapping to domain names."}
    )
    domain2idx: dict = field(
        default_factory=dict, 
        metadata={"help": "domain names mapping to indices."}
    )
    train_ids: torch.Tensor = field(
        default=None, 
        metadata={"help": "Training domain indices."}
    )
    tgt_ids: torch.Tensor = field(
        default=None, 
        metadata={"help": "Target domain indices."}
    )
    curriculum_path: str = field(
        default=None, metadata={"help": "Path to stage-wise curriculum domain weights (.pkl)."}
    )

def domain_gen(data, seq_len, domain_id=None):
    if domain_id is None:
        for i in range(len(data)//seq_len):
            yield {"input_ids": data[i*seq_len:(i+1)*seq_len]}
    else:
        for i in range(len(data)//seq_len):
            yield {"domain_ids": torch.tensor([domain_id], dtype=torch.long), "input_ids": data[i*seq_len:(i+1)*seq_len]}

class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):

    def __init__(self, ex_iterables, generator, probabilities=None, probabilities_handle=None, stopping_strategy="all_exhausted",
                 curriculum_dict=None):
        '''
        probabilities: vector of static probabilities over training
        probabilities_handle: handle to domain weights buffer in model params
        '''
        super().__init__(ex_iterables, generator, stopping_strategy=stopping_strategy)
        self.probabilities_handle = probabilities_handle
        self.probabilities = probabilities
        self.curriculum_dict = curriculum_dict
        if curriculum_dict is not None:
            self.step = 0
            
    @staticmethod
    def _iter_random_indices(rng, num_sources, probabilities_handle=None, p=None, random_batch_size=RANDOM_BATCH_SIZE):
        while True:
            # read domain weights
            if probabilities_handle is not None:
                p = probabilities_handle.detach().cpu().numpy()
            yield from WeightedRandomSampler(weights=p, num_samples=random_batch_size, replacement=True)

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        return self._iter_random_indices(rng, len(self.ex_iterables), probabilities_handle=self.probabilities_handle, probabilities=self.probabilities)

    def shard_data_sources(self, shard_indices):
        return self

    @property
    def n_shards(self):
        return 1

    def shuffle_data_sources(self, seed):
        self.ex_iterables = [ex_iterable.shuffle_data_sources(seed) for ex_iterable in self.ex_iterables]
        return self


def interleave_datasets(datasets, probabilities=None, probabilities_handle=None, seed=None, stopping_strategy='all_exhausted'):
    iterable_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)
    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator,
            probabilities=probabilities, probabilities_handle=probabilities_handle,
            stopping_strategy=stopping_strategy)

    return IterableDataset(ex_iterable=ex_iterable)

def get_train_eval_datasets(data_config:DataTrainingArguments,
                            verbose:bool=False,
                            doremi:bool=False,
                            **kwargs):
    data_dict = get_dataset(data_config)
    if 'all' in data_dict['train'].keys():
        del data_dict['train']['all']
    if 'all' in data_dict['val'].keys():
        del data_dict['val']['all']
    if doremi and ('mix' in data_dict['train'].keys()):
        del data_dict['train']['mix']
        del data_dict['val']['mix']
        
    seed = 42
    sequence_length = data_config.max_token_length
    max_train_samples = data_config.max_train_samples
    max_eval_samples = data_config.max_eval_samples

    domain_list = list(data_dict['train'].keys())
    idx2domain = {i:dom for i,dom in enumerate(domain_list)}
    domain2idx = {dom:i for i,dom in idx2domain.items()}
    train_ids = torch.tensor([domain2idx[name] for name in data_config.train_domains.split(',')])
    tgt_ids = torch.tensor([domain2idx[name] for name in data_config.tgt_domains.split(',')])
    
    all_domain_ids = torch.concat([train_ids, tgt_ids]).numpy()
    curriculum_dict = None
    if data_config.curriculum_path is not None:
        with open(data_config.curriculum_path, "rb") as trg:
            curriculum_dict = pickle.load(trg)
            
    if curriculum_dict is not None:
        train_dw = torch.tensor(curriculum_dict[0], dtype=torch.float)   
    elif data_config.train_dw is None:
        train_dw = torch.ones(len(domain_list), dtype=torch.float)/len(set(all_domain_ids))
        if len(domain_list)>len(set(all_domain_ids)):
            exclude_ids = torch.tensor([i for i in range(len(domain_list)) if torch.tensor(i) not in all_domain_ids])
            train_dw[exclude_ids] = 0.0
    else:
        train_dw = torch.tensor([float(i) for i in data_config.train_dw.split(",")])
        
    if data_config.val_dw is None:
        if 'mix' not in data_config.tgt_domains:
            val_dw = torch.zeros(len(domain_list), dtype=torch.float)
            val_dw[tgt_ids] = 1/len(tgt_ids)
        else:
            val_dw = torch.ones(len(domain_list), dtype=torch.float)/len(domain_list)
    else:
        val_dw = torch.tensor([float(i) for i in data_config.val_dw.split(",")])
    
    domain_config = DomainConfigArguments(
                    domain_list=domain_list,
                    idx2domain=idx2domain,
                    domain2idx=domain2idx,
                    train_ids=train_ids,
                    tgt_ids=tgt_ids,
                    train_dw=train_dw,
                    val_dw=val_dw,
                    curriculum_path=data_config.curriculum_path,
                    **kwargs)
    train_dict = {domain2idx[dom]:v for dom,v in data_dict['train'].items()}
    val_dict = {domain2idx[dom]:v for dom,v in data_dict['val'].items() if val_dw[domain2idx[dom]]>0}
    
    train_dataset_ls, val_dataset_ls = [], []
    for k in train_dict.keys():
        train_domain_dataset = IterableDataset.from_generator(domain_gen,
                                                gen_kwargs={'data': train_dict[k],
                                                            'seq_len': sequence_length,
                                                            'domain_id': k,
                                                            }
                                                )
        train_dataset_ls.append(train_domain_dataset)
        if verbose:
            print(f'{idx2domain[k]} loaded!')
    
    val_dw_gen = []
    for k in val_dict.keys():
        val_domain_dataset = IterableDataset.from_generator(domain_gen,
                                                gen_kwargs={'data': val_dict[k],
                                                            'seq_len': sequence_length,
                                                            'domain_id': k,
                                                            }
                                                )
        val_dataset_ls.append(val_domain_dataset)
        val_dw_gen.append(val_dw[k])
        
    train_ds = interleave_datasets(
                    train_dataset_ls,
                    probabilities=train_dw,
                    probabilities_handle=train_dw,
                    seed=seed)
    val_ds = interleave_datasets(
                    val_dataset_ls,
                    probabilities=torch.tensor(val_dw_gen),
                    probabilities_handle=torch.tensor(val_dw_gen),
                    seed=seed)
    
    
    def take_data_generator(ds, max_samples):
        idx = 0
        for ex in ds:
            yield ex
            idx += 1
            if max_samples is not None and idx >= max_samples:
                return
    if max_train_samples is not None:
        train_ds = IterableDataset.from_generator(take_data_generator, gen_kwargs={'ds': train_ds, 'max_samples': max_train_samples})
    if max_eval_samples is not None:
        val_ds = IterableDataset.from_generator(take_data_generator, gen_kwargs={'ds': val_ds, 'max_samples': max_eval_samples})
    if 'wiki40b' in data_config.dataset:
        tokenizer = AutoTokenizer.from_pretrained('/scratch/pagliard/doge/exp/doge_frozen_weights_12l_catalan-mu0001_seed42_10k/checkpoint-10000')
    elif 'cfg' in data_config.dataset:
        tokenizer = CFGTokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if 'cfg' not in data_config.dataset:
        tokenizer.model_max_length=data_config.max_token_length
    if data_config.curriculum_path is not None:
        return train_ds, val_ds, domain_config, tokenizer, train_dataset_ls
    else:
        return train_ds, val_ds, domain_config, tokenizer, None

def get_data_collator(tokenizer, return_tensors='pt', do_padding=False, max_length=1024):
    def data_collator(features):
        if not do_padding:
            try:
                batch = {
                        k: torch.tensor([f[k] for f in features])
                        for k in features[0].keys() if k!='input_ids'
                        }
                if not torch.is_tensor(batch['input_ids']):
                    batch['input_ids'] = torch.tensor([np.array(f['input_ids'], dtype=np.int32) for f in features])
            except Exception:
                batch = {
                        k: torch.tensor([np.array(f[k], dtype=np.int32) for f in features])
                        for k in features[0].keys()
                        }
        else:
            try:
                batch = tokenizer.pad(features, return_tensors=return_tensors, pad_to_multiple_of=max_length)
            except:
                raise Exception
        batch['input_ids'] = batch['input_ids'].long()
        if 'attention_mask' not in batch:
            batch['attention_mask'] = torch.ones_like(batch['input_ids']).long()
        else:
            batch['attention_mask'] = batch['attention_mask'].long()

        batch.pop("special_tokens_mask", None)
        if 'labels' not in batch:
            labels = batch['input_ids'].clone()
            batch["labels"] = labels

        try:
            if tokenizer.pad_token_id is not None:
                batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        except:
            pass

        if 'domain_ids' not in batch and 'domain_id' in batch:
            batch['domain_ids'] = batch['domain_id']  # compat
            batch.pop('domain_id')
        # print(batch)
        return batch
    return data_collator
