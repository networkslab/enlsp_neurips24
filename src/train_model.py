from typing import List
import os
import transformers    
from transformers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer, IntervalStrategy

import torch

from datasets import load_dataset, Split, load_from_disk, DatasetDict
import argparse
from datetime import datetime

import numpy as np
import copy

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, PropagationMode
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM
from src.utils.utils import get_abs_path
from src.utils.train_utils import SFTConfigGenerate, SFTTrainerGenerate, DataCollatorForSeq2SeqGenerate, fix_the_seed
import src.utils.train_utils as train_utils
from src.utils.training_args import SAVED_ARGS

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_args", type=str, default='full_prop_args')
    parser_args = parser.parse_args()
    if parser_args.training_args not in SAVED_ARGS:
        raise ValueError(f"Training args {parser_args.training_args} not found in SAVED_ARGS")
    args = SAVED_ARGS[parser_args.training_args]
    validate_args(args)
    
    fix_the_seed(SEED)

    transformers.logging.set_verbosity_info()
    if args.ddp:
        torch.distributed.init_process_group("nccl")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        rank = os.environ['LOCAL_RANK'] #rank when using DDP
        deepspeed = get_abs_path(['src','utils'])+ args.deepspeed if args.deepspeed is not None else None
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        rank = 0
        deepspeed = None

    model_name = args.model
    dataset_name = args.dataset
    
    #tokenizer
    sep_token = AddedToken("<SEP>", lstrip=False, rstrip=False)
    # TODO update model to account for expanded vocab
    tokenizer = AutoTokenizer.from_pretrained(
        get_abs_path([model_name]) if args.load_model_from_disk else model_name, 
        padding_side='left', use_fast=False, 
        sep_token=sep_token
        )
    
    #templates
    instruction_template_ids = tokenizer(args.instruction_template,add_special_tokens=False) ['input_ids'] + [tokenizer.sep_token_id]
    response_template_ids = tokenizer(args.response_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]

    #Dataset
    if args.load_dataset_from_disk:
        tokenized_dataset = load_from_disk(get_abs_path(['data','datasets',args.dataset]))
    
    else:
        full_dataset = load_dataset(dataset_name, split=Split.TRAIN)
        #full_dataset = full_dataset.select(indices=range(200))
        dataset = full_dataset.train_test_split(test_size=0.2, seed=SEED) 
        tokenized_dataset_train, tokenized_dataset_val = train_utils.tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids)
        tokenized_dataset = DatasetDict({'train': tokenized_dataset_train, 'validation': tokenized_dataset_val})
    
        if args.save_dataset_dir is not None and rank == 0:
            tokenized_dataset.save_to_disk(get_abs_path(['data','datasets',args.save_dataset_dir]))
        
        
  
    #DataCollator
    collator = DataCollatorForSeq2SeqGenerate(tokenizer=tokenizer)

    #Model
    if args.load_model_from_disk:
        adalas_config = AdalasOPTConfig.from_pretrained(get_abs_path([model_name]))
        adalas = AdalasOPTForCausalLM.from_pretrained(get_abs_path([model_name]),config=adalas_config)
        print(f"Loading model from {model_name}. Model config parameters will be ignored")
    else:
        propagation_config = args.prop_config
        adalas_config = AdalasOPTConfig.from_pretrained(model_name)
        adalas_config.propagation_config = propagation_config
        adalas_config.skip_prompt = args.skip_prompt
        adalas_config.sep_token_id = tokenizer.sep_token_id
        adalas = AdalasOPTForCausalLM.from_pretrained(model_name,config=adalas_config)
        
    if args.fp16:
        adalas = adalas.to(torch.float16)
    
    if args.save_model_pretrain_dir is not None and rank == 0:
        tokenizer.save_pretrained(get_abs_path(['results','pre_train',args.save_model_pretrain_dir]))
        adalas.save_pretrained(get_abs_path(['results','pre_train',args.save_model_pretrain_dir]))

    stripped_model_name = model_name.split('/')[-1]
    stripped_dataset_name = dataset_name.split('/')[-1]
    if args.ddp:
        torch.distributed.barrier()
    time = datetime.now()
    current_time_str = time.strftime("%d-%m_%H-%M-%S")
    output_dir_name = f'{stripped_model_name}/{stripped_dataset_name}_{current_time_str}'

    #Metrics
    def compute_metrics(eval_pred):
        return train_utils.compute_metrics(eval_pred, tokenizer)
   
    #Training
    sft_config = SFTConfigGenerate(
        learning_rate = args.learning_rate,
        packing=False, 
        output_dir=get_abs_path(['results', output_dir_name]),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        gradient_checkpointing = args.gradient_checkpointing,
        num_train_epochs=args.train_epochs,
        max_seq_length=args.max_seq_length,
        report_to=['tensorboard'], 
        logging_steps=args.logging_steps, 
        logging_dir=get_abs_path(['logs', output_dir_name]),
        logging_first_step=True,
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.eval_steps, 
        save_strategy=args.save_strategy,
        include_inputs_for_metrics=True,
        eval_with_generate=True,
        max_new_tokens=args.max_new_tokens,
        deepspeed=deepspeed,
        local_rank=rank if args.ddp else None,
        ddp_find_unused_parameters=False,
        )
    trainer = SFTTrainerGenerate(
        model=adalas,
        args=sft_config,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.neftune_noise_alpha = None # temporary fix https://github.com/huggingface/trl/issues/1837
    trainer.train()

def validate_args(args):
    if args.prop_config.propagation_mode == PropagationMode.STATIC_SKIP:
        assert len(args.prop_config.skip_layers) > 0, "STATIC SKIP needs a list of layers to skip"
    if args.prop_config.propagation_mode == PropagationMode.STOCHASTIC_DROPOUT:
        assert len(args.prop_config.skip_probs) > 0, 'STOCHASTIC DROPOUT needs a list of skip probabilities'

if __name__ == "__main__":
    main()