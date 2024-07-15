from typing import List
import os
from transformers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer, IntervalStrategy

from datasets import load_dataset, Split, load_from_disk
import argparse

import numpy as np
import copy

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, PropagationMode
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM
from src.utils.utils import get_abs_path
from src.utils.train_utils import SFTConfigGenerate, SFTTrainerGenerate, DataCollatorForCompletionOnlyLMGenerate
import src.utils.train_utils as train_utils
from src.utils.training_args import SAVED_ARGS

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_args", type=str, default='full_prop_args')
    parser_args = parser.parse_args()
    if parser_args.training_args not in SAVED_ARGS:
        raise ValueError(f"Training args {parser_args.training_args} not found in SAVED_ARGS")
    args = SAVED_ARGS[parser_args.training_args]
    validate_args(args)

    MODEL_NAME = args.model
    DATASET_NAME = args.dataset
    
    #tokenizer
    sep_token = AddedToken("<SEP>", lstrip=False, rstrip=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left', use_fast=False, sep_token=sep_token)
    
    #templates
    instruction_template_ids = tokenizer(args.instruction_template,add_special_tokens=False) ['input_ids'] + [tokenizer.sep_token_id]
    response_template_ids = tokenizer(args.response_template,add_special_tokens=False)['input_ids'] + [tokenizer.sep_token_id]

    #Dataset
    if args.load_dataset_from_disk:
        tokenized_dataset = load_from_disk(get_abs_path(['data','saved_datasets',args.dataset]))
        tokenized_dataset_train = tokenized_dataset['train']
        tokenized_dataset_val = tokenized_dataset['validation']
    
    else:
        full_dataset = load_dataset(DATASET_NAME, split=Split.TRAIN)
        dataset = full_dataset.train_test_split(test_size=0.2)
        

        
        tokenized_dataset_train, tokenized_dataset_val = train_utils.tokenize_and_format_dataset(dataset, DATASET_NAME, tokenizer, args, instruction_template_ids, response_template_ids)
        
  
    #DataCollator
    collator = DataCollatorForCompletionOnlyLMGenerate(instruction_template=instruction_template_ids, response_template=response_template_ids, tokenizer=tokenizer, mlm=False)

    #Model
    propagation_config = args.prop_config
    adalas_config = AdalasOPTConfig.from_pretrained(MODEL_NAME)
    adalas_config.propagation_config = propagation_config
    adalas = AdalasOPTForCausalLM.from_pretrained(MODEL_NAME, config=adalas_config)

    stripped_model_name = MODEL_NAME.split('/')[-1]
    stripped_dataset_name = DATASET_NAME.split('/')[-1]
    output_dir_name = f'{stripped_model_name}/{stripped_dataset_name}'

    #Metrics
    def compute_metrics(eval_pred):
        train_utils.compute_metrics(eval_pred, tokenizer)

    #Training
    sft_config = SFTConfigGenerate(
        packing=False, 
        output_dir=get_abs_path(['logs', output_dir_name]),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.train_epochs,
        max_seq_length=args.max_seq_length,
        report_to=['tensorboard'], 
        logging_steps=20, 
        logging_dir=get_abs_path(['logs', output_dir_name]),
        logging_first_step=True,
        evaluation_strategy='steps', 
        eval_steps=2, 
        save_strategy=IntervalStrategy.NO,
        include_inputs_for_metrics=True,
        eval_with_generate=True
        )
    trainer = SFTTrainerGenerate(
        model=adalas,
        args=sft_config,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

def validate_args(args):
    if args.prop_mode == PropagationMode.STATIC_SKIP:
        assert len(args.skip_layers) > 0, "STATIC SKIP needs a list of layers to skip"
    if args.prop_mode == PropagationMode.STOCHASTIC_DROPOUT:
        assert len(args.skip_probs) > 0, 'STOCHASTIC DROPOUT needs a list of skip probabilities'

if __name__ == "__main__":
    main()