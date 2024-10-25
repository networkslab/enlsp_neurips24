from typing import List
import os
import transformers
from transformers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer, IntervalStrategy
from peft import LoraConfig, TaskType, get_peft_model, LoftQConfig

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import load_from_disk
from datetime import datetime

from time import sleep

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, PropagationMode
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM
from src.utils.utils import get_abs_path, fix_the_seed, get_args, freeze_top_decoder_layers, freeze_skipped_decoder_layers
from src.utils.train_utils import DataCollatorForSeq2SeqGenerate
from src.training.sft_trainer_generate import SFTTrainerGenerate, SFTConfigGenerate
import src.utils.train_utils as train_utils
from src.utils.train_utils import DATASET_KEYS



def main():
    args = get_args()
    
    fix_the_seed(args.seed)

    transformers.logging.set_verbosity_info()
    if args.ddp:
        torch.distributed.init_process_group("nccl")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        rank = int(os.environ['LOCAL_RANK']) #rank when using DDP
        deepspeed = get_abs_path(['src','utils'])+ args.deepspeed if args.deepspeed is not None else None
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        rank = 0
        deepspeed = None


    model_name = args.model
    dataset_name = args.dataset

    #tokenizer
    sep_token = AddedToken("<SEP>", lstrip=False, rstrip=False)
    tokenizer = AutoTokenizer.from_pretrained(
        get_abs_path([model_name]) if args.load_model_from_disk else model_name,
        padding_side='left', use_fast=False,
        sep_token=sep_token
    )
    
    #Dataset
    if args.tokenized_dataset_path is not None:
        tokenized_dataset = load_from_disk(get_abs_path(['data','datasets',args.tokenized_dataset_path]))

    else:
        tokenized_dataset = DATASET_KEYS[dataset_name]["prepare_fnc"](tokenizer, args)

        if args.save_dataset_dir is not None and rank == 0:
            tokenized_dataset.save_to_disk(get_abs_path(['data','datasets',args.save_dataset_dir]))
        
    #DataCollator
    collator = DataCollatorForSeq2SeqGenerate(tokenizer=tokenizer)

    #sleep to stagger the model loading, in order to avoid high peak RAM use
    sleep(rank*5)
    print(f'rank {rank} starting model loading')

    #Model
    if args.load_model_from_disk:
        adalas_config = AdalasOPTConfig.from_pretrained(get_abs_path([model_name]))
        adalas_config.propagation_config = args.prop_config
        adalas_config.with_cost_aware_loss = args.with_cost_aware_loss
        adalas_config.alpha = args.alpha
        adalas = AdalasOPTForCausalLM.from_pretrained(get_abs_path([model_name]),config=adalas_config)
        print(f'Alpha: {args.alpha}')
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

    #adalas.freeze_backbone(freeze_head=False) Do not freeze backbone
    if args.save_model_pretrain_dir is not None and rank == 0:
        tokenizer.save_pretrained(get_abs_path(['results','pre_train',args.save_model_pretrain_dir]))
        adalas.save_pretrained(get_abs_path(['results','pre_train',args.save_model_pretrain_dir]))
    if args.with_lora:
        lora_conf = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                               lora_dropout=args.lora_dropout, task_type=TaskType.CAUSAL_LM,
                               target_modules=['k_proj', 'v_proj', 'q_proj', 'lm_head'])

        adalas = get_peft_model(adalas, lora_conf)

    if args.load_model_from_disk:
        stripped_model_name = model_name.split('/')[1]
    else:
        stripped_model_name = model_name.split('/')[-1]
    stripped_dataset_name = dataset_name.split('/')[-1]
    if args.ddp:
        torch.distributed.barrier()
    time = datetime.now()
    current_time_str = time.strftime("%d-%m_%H-%M-%S")
    output_dir_name = f'{stripped_model_name}/{stripped_dataset_name}_{current_time_str}'
    
    if adalas_config.propagation_config.propagation_mode == PropagationMode.STATIC_EE and adalas_config.propagation_config.freeze_subsequent:
        freeze_top_decoder_layers(adalas, ['lm_head'],
                                  last_unfrozen_layer=adalas_config.propagation_config.early_exit_layer, verbose=True)
        
    if adalas_config.propagation_config.propagation_mode == PropagationMode.STATIC_SKIP and adalas_config.propagation_config.freeze_skipped:
        freeze_skipped_decoder_layers(adalas, ['lm_head'], adalas_config.propagation_config.skip_layers, verbose=True)
    # Metrics
    def compute_metrics(eval_pred):
        return train_utils.compute_metrics(eval_pred, tokenizer)

    # Training
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
        evaluation_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        include_inputs_for_metrics=True,
        eval_with_generate=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.generate_do_sample,
        temperature=args.generate_temperature,
        deepspeed=deepspeed,
        local_rank=rank if args.ddp else None,
        ddp_find_unused_parameters=False,
    )
    
    summary_writer = SummaryWriter(sft_config.logging_dir + '/custom_scalars')
    summary_writer.add_custom_scalars(train_utils.get_tensorboard_training_layout(adalas.model.decoder))
    metrics_callback = train_utils.MetricsCallback(summary_writer, adalas.model.decoder)
    trainer = SFTTrainerGenerate(
        model=adalas,
        args=sft_config,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )
    trainer.neftune_noise_alpha = None # temporary fix due to HF bug
    trainer.train() # make sure there are trainable components, otherwise the backprop will fail.
    # trainer.evaluate()

if __name__ == "__main__":
    main()
