#!/usr/bin/python3
"""
Train a tokenizer on a subset of languages of wiki40B

pip install -q tfds-nightly tensorflow tqdm transformers
"""

import argparse
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import numpy as np
import os


end_of_doc_token = '</s>'


args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--langs', default='en,de,fr,ru,es,nl', type=str)
args_parser.add_argument('--weights', default='0.1,0.1,0.1,0.1,0.1,1', type=str) # subsampling large language dataset
args_parser.add_argument('--data_path', default='path to wiki40b data', type=str)
args_parser.add_argument('--vocab_size', default=52000, type=int)


def get_text(doc):
    text = doc['text'].numpy().decode('utf-8')
    tokens = text + ' ' + end_of_doc_token
    return tokens


def get_training_corpus(languages, weights):
    for lang, weight in zip(languages, weights):
        dataset, dataset_info = tfds.load(f"wiki40b/{lang}", with_info=True, data_dir='./tensorflow_datasets')
        for split in ['test', 'validation', 'train']:

            print(f"Size of the {split} set for {lang}: {len(dataset[split])} documents")
            dataset_list = list(dataset[split])

            for doc in tqdm(dataset_list, desc=f"tokenizing {split} for {lang}"):
                if random.random() < weight:
                    yield get_text(doc)


def process_document(doc, tokenizer):
    text = doc['text'].numpy().decode('utf-8')
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids


def tokenize_dataset(languages, tokenizer, data_path, tokenizer_name):

    for lang in languages:
        dataset, dataset_info = tfds.load(f"wiki40b/{lang}", with_info=True)
        os.makedirs(f"{data_path}/{tokenizer_name}_{lang}", exist_ok=True)
        for split in ['test', 'validation', 'train']:

            filename = os.path.join(f"{data_path}/{tokenizer_name}_{lang}/", f'{lang}_{split}.bin')
            if os.path.exists(filename):
                print(f"{filename} already exist, skipping ...")
                continue

            print(f"Size of the {split} set for {lang}: {len(dataset[split])} documents")

            dataset_list = list(dataset[split])

            results = []
            for doc in tqdm(dataset_list, desc=f"tokenizing {split} for {lang}"):
                results.append(process_document(doc, tokenizer))

            print(f"Processed {len(results)} documents in {split} set for {lang}")

            arr_len = np.sum([len(seq) for seq in results])
            dtype = np.uint32
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            idx = 0
            for start in tqdm(range(0, len(results), 1024), desc=f'writing {filename}'):
                arr_batch = np.array([x for y in results[start:start+1024] for x in y], dtype=np.uint32)
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()



if __name__ == "__main__":

    args = args_parser.parse_args()

    languages = args.langs.split(',')
    weights = [float(x) for x in args.weights.split(',')]
    if len(weights) == 1:
        weights = [weights[0]] * len(languages)

    assert len(languages) == len(weights)

    print(languages)
    print(weights)

    lang_str = ','.join(sorted(languages))
    tokenizer_name = f"tokenizer-{lang_str}"

    if not os.path.exists(f"{tokenizer_name}.tknzr"):

        old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(languages, weights), args.vocab_size)
        tokenizer.save_pretrained(f"{tokenizer_name}.tknzr")
    
    else:

        tokenizer = AutoTokenizer.from_pretrained(f"./tokenizer-{lang_str}.tknzr")

    tokenize_dataset(languages, tokenizer, args.data_path, tokenizer_name)
