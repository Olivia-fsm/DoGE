import os
from transformers import XLMTokenizer, AutoTokenizer
import numpy as np
import os
import torch
from datasets import load_dataset, Dataset
import tensorflow_datasets as tfds


tknzr = AutoTokenizer.from_pretrained('/scratch/homes/sfan/models/doge_codebase/src/data/tknzr/tokenizer-ca,de,en,es,fr,ru.tknzr')
end_of_doc_token = '</s>'
languages = ['en', 'ar', 'zh-cn', 'zh-tw', 'nl', 'fr', 'de', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'es', 'th', 'tr', 'bg', 'ca', 'cs', 'da', 'el', 'et', 'fa', 'fi', 'he', 'hi', 'hr', 'hu', 'id', 'lt', 'lv', 'ms', 'no', 'ro', 'sk', 'sl', 'sr', 'sv', 'tl', 'uk', 'vi']


def get_wiki40b(subset='en', num_proc=40,
                return_torch=True):
    """ https://huggingface.co/datasets/wiki40b
    """
    WIKI_40B_PATH = os.path.join(os.path.dirname(__file__), "wiki40b")
    SUBSET_PATH = os.path.join(WIKI_40B_PATH, subset)
    train_path = os.path.join(SUBSET_PATH, f"{subset}_train.bin")
    test_path = os.path.join(SUBSET_PATH, f"{subset}_test.bin")
    
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    test_data = np.memmap(test_path, dtype=np.uint16, mode='r')
    print(f'Subset {subset}: train[{len(train_data)}] | val[{len(test_data)}]')
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        test_data = torch.tensor(np.array(test_data, dtype=np.int32))
    return {'train': train_data, 'val': test_data, 'test': test_data}

get_wiki40b(subset='ca', num_proc=10,
                return_torch=False)