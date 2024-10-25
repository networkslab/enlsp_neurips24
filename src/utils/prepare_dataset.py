from datasets import load_dataset, DatasetDict, Split
from src.utils import train_utils

def prepare_samsum(tokenizer, args):
    dataset_name = "Samsung/samsum"
    instruction_template = "### Dialogue:"
    response_template= "\n### Summary:"
    
    instruction_template_ids = tokenizer(instruction_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]
    response_template_ids = tokenizer(response_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]

    dataset = load_dataset(dataset_name)
    # dataset['train'] = dataset['train'].select(indices=range(200))
    # dataset['validation'] = dataset['validation'].select(indices=range(200))
    # dataset['test'] = dataset['test'].select(indices=range(200))
    tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test = train_utils.tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids)
    tokenized_dataset = DatasetDict({'train': tokenized_dataset_train, 'validation': tokenized_dataset_val, 'test': tokenized_dataset_test})
    return tokenized_dataset

def prepare_reddit():
    pass

def prepare_cnndm(tokenizer, args):
    dataset_name = "abisee/cnn_dailymail"
    instruction_template= "### Article:"
    response_template= "\n### Summary:"
    
    instruction_template_ids = tokenizer(instruction_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]
    response_template_ids = tokenizer(response_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]
    
    dataset_version = '3.0.0'
    dataset = load_dataset(dataset_name,dataset_version)
    tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test = train_utils.tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids)
    tokenized_dataset = DatasetDict({'train': tokenized_dataset_train, 'validation': tokenized_dataset_val, 'test': tokenized_dataset_test})
    return tokenized_dataset

def prepare_alpaca(tokenizer, args):
    dataset_name = "tatsu-lab/alpaca"
    instruction_template= "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:"
    response_template= "\n### Response:"
    context_template="\n### Input:"
    
    instruction_template_ids = tokenizer(instruction_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]
    response_template_ids = tokenizer(response_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]
    
    full_dataset = load_dataset(dataset_name, split=Split.TRAIN)
    # full_dataset = full_dataset.select(indices=range(300))
    dataset = full_dataset.train_test_split(test_size=0.3,seed=args.seed)
    
    dataset_val_test = dataset['test'].train_test_split(test_size=0.5,seed=args.seed)
    dataset['validation'] = dataset_val_test['train']
    dataset['test'] = dataset_val_test['test']
    tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test = train_utils.tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids, context_template)
    tokenized_dataset = DatasetDict({'train': tokenized_dataset_train, 'validation': tokenized_dataset_val, 'test': tokenized_dataset_test})
    return tokenized_dataset
    