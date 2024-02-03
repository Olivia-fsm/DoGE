import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 
import os
import torch
tknzr = tiktoken.get_encoding("gpt2")

SUBSET2META = {
    'arxiv': 'RedPajamaArXiv',
    'book': 'RedPajamaBook',
    'cc': 'RedPajamaCommonCrawl',
    'c4': 'RedPajamaC4',
    'github': 'RedPajamaGithub',
    'stackexchange': 'RedPajamaStackExchange',
    'wikipedia': 'RedPajamaWikipedia',
    
}

def get_slimpajama(subset='arxiv', num_proc=40,
                       return_torch=False,):
    """ Full: https://huggingface.co/datasets/cerebras/SlimPajama-627B
        6B-subset: DKYoon/SlimPajama-6B
    """
    # {
    #     "text": ...,
    #     "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...},
    #     "red_pajama_subset": "common_crawl" | "c4" | "github" | "books" | "arxiv" | "wikipedia" | "stackexchange"
    # }
    SLIM_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/slim_redpajama/")
    SUBSET_PATH = os.path.join(SLIM_DATA_PATH, subset)
    subset_name = SUBSET2META[subset]
    if not os.path.exists(os.path.join(SUBSET_PATH, 'val.bin')):
        os.makedirs(SUBSET_PATH, exist_ok=True)
        dataset = load_dataset("cerebras/SlimPajama-627B", split=['train', 'test'])
        data_dict = {}
        data_dict['train'] = dataset[0].filter(lambda example: example["meta"]['redpajama_set_name']==subset_name)
        data_dict['val'] = dataset[1].filter(lambda example: example["meta"]['redpajama_set_name']==subset_name)
        
        def process(example):
            ids = tknzr.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(tknzr.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = {}
        tokenized['train'] = data_dict['train'].map(
            process,
            remove_columns=['text', 'meta'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        tokenized['val'] = data_dict['val'].map(
            process,
            remove_columns=['text', 'meta'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(SUBSET_PATH, f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 100

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(os.path.join(SUBSET_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(SUBSET_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    print(f'Subset {subset}: train[{len(train_data)}] | val[{len(val_data)}]')
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        val_data = torch.tensor(np.array(val_data, dtype=np.int32))
    return {'train': train_data, 'val': val_data}


def get_slimpajama_6b(subset='arxiv', num_proc=40,
                          return_torch=False):
    """ Full: https://huggingface.co/datasets/cerebras/SlimPajama-627B
        6B-subset: DKYoon/SlimPajama-6B
    """
    # {
    #     "text": ...,
    #     "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...},
    #     "red_pajama_subset": "common_crawl" | "c4" | "github" | "books" | "arxiv" | "wikipedia" | "stackexchange"
    # }
    REDPAJIMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/slim_6b/")
    SUBSET_PATH = os.path.join(REDPAJIMA_DATA_PATH, subset)
    subset_name = SUBSET2META[subset]
    print('Load subset_name: ', subset_name)
    if not os.path.exists(os.path.join(SUBSET_PATH, 'val.bin')):
        os.makedirs(SUBSET_PATH, exist_ok=True)
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=['train', 'test'])
        print(dataset)
        data_dict = {}
        data_dict['train'] = dataset[0].filter(lambda example: example["meta"]['redpajama_set_name']==subset_name)
        data_dict['val'] = dataset[1].filter(lambda example: example["meta"]['redpajama_set_name']==subset_name)

        print(data_dict['train'])
        def process(example):
            'Processing dataset...'
            ids = tknzr.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(tknzr.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = {}
        tokenized['train'] = data_dict['train'].map(
            process,
            remove_columns=['text', 'meta', '__index_level_0__'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        tokenized['val'] = data_dict['val'].map(
            process,
            remove_columns=['text', 'meta', '__index_level_0__'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            print('Columns: ', dset.features)
            arr_len = np.sum(dset['len'])
            filename = os.path.join(SUBSET_PATH, f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(os.path.join(SUBSET_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(SUBSET_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        val_data = torch.tensor(np.array(val_data, dtype=np.int32))
    return {'train': train_data, 'val': val_data}
