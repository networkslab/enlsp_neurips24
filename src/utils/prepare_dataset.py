from datasets import load_dataset, DatasetDict, Split
from src.utils import train_utils


def prepare_databricks(tokenizer, args, instruction_template_ids, response_template_ids):
    dataset_name = "databricks/databricks-dolly-15k"
    full_dataset = load_dataset(dataset_name, split=Split.TRAIN)
    dataset = full_dataset.train_test_split(test_size=0.2,seed=args.seed)
    dataset['validation'] = dataset['test']
    del dataset['test']
    tokenized_dataset_train, tokenized_dataset_val = train_utils.tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids)
    tokenized_dataset = DatasetDict({'train': tokenized_dataset_train, 'validation': tokenized_dataset_val})
    return tokenized_dataset

def prepare_samsum(tokenizer, args, instruction_template_ids, response_template_ids):
    dataset_name = "Samsung/samsum"
    dataset = load_dataset(dataset_name)
    tokenized_dataset_train, tokenized_dataset_val = train_utils.tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids)
    tokenized_dataset = DatasetDict({'train': tokenized_dataset_train, 'validation': tokenized_dataset_val})
    return tokenized_dataset

def prepare_reddit():
    pass

def prepare_cnndm():
    pass