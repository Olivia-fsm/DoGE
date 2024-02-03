import tensorflow_datasets as tfds
from transformers import XLMTokenizer, AutoTokenizer
import numpy as np
import concurrent.futures
import os
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('src/data/tknzr/tokenizer-ca,de,en,es,fr,ru.tknzr')
end_of_doc_token = '</s>'

# languages = ['en', 'ar', 'zh-cn', 'zh-tw', 'nl', 'fr', 'de', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'es', 'th', 'tr', 'bg', 'ca', 'cs', 'da', 'el', 'et', 'fa', 'fi', 'he', 'hi', 'hr', 'hu', 'id', 'lt', 'lv', 'ms', 'no', 'ro', 'sk', 'sl', 'sr', 'sv', 'tl', 'uk', 'vi']
languages = ['en', 'fr', 'de', 'ru', 'es', 'nl', 'ca']


def process_document(doc):
    text = doc['text'].numpy().decode('utf-8')
    tokens = tokenizer.tokenize(text) + [end_of_doc_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

WIKI40B_PATH = os.path.join(os.path.dirname(__file__), "datasets", "wiki40b")
for lang in languages:
    dataset, dataset_info = tfds.load(f"wiki40b/{lang}", with_info=True)
    os.makedirs(os.path.join(WIKI40B_PATH, f"{lang}"), exist_ok=True)
    for split in ['test', 'train']:

        filename = os.path.join(WIKI40B_PATH, f"{lang}", f'{lang}_{split}.bin')
        if os.path.exists(filename):
            print(f"{filename} already exist, skipping ...")
            continue

        print(f"Size of the {split} set for {lang}: {len(dataset[split])} documents")

        dataset_list = list(dataset[split])

        results = []
        for doc in tqdm(dataset_list, desc=f"tokenizing {split} for {lang}"):
            results.append(process_document(doc))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        #     # Submit all tasks to the executor
        #     futures = [executor.submit(process_document, doc) for doc in dataset_list]
        #     # Create a tqdm progress bar
        #     results = []
        #     with tqdm(total=len(futures), desc=f"tokenizing {split} for {lang}") as progress:
        #         for future in concurrent.futures.as_completed(futures):
        #             result = future.result()
        #             results.append(result)
        #             # Update progress bar for each completed task
        #             progress.update(1)

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
