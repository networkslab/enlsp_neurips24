
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.trainer_utils import EvaluationStrategy

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, \
    StochasticDropoutPropagationConfig, PropagationConfig, PropagationMode, DynamicPropagationConfig

class DictOverwritable(object):
    '''allows to overwrite some attributes of a class with a dict'''
    def update_fields(self, dict_for_update):
        for key, value in dict_for_update.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class TrainingArgs(DictOverwritable):
    learning_rate: float = 5e-5
    load_dataset_from_disk: bool = False
    load_model_from_disk: bool = False
    dataset: str = 'databricks/databricks-dolly-15k'
    model: str = 'facebook/opt-125M'
    prop_config: PropagationConfig = PropagationConfig()
    batch_size: int = 10
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    train_epochs: int = 3
    max_seq_length: int = 768
    logging_steps:int = 20
    eval_strategy: str = "steps"
    eval_steps:int = 100
    save_steps:int=100
    save_strategy: str = "no"
    prompt_seq_length: float = 0.7
    from_checkpoint: bool = False
    load_best_model_at_end: bool = False,
    save_total_limit: int = 1,
    multiprocess: bool = True
    instruction_template: str = "### User:"
    response_template: str = "\n### Assistant:"
    ddp: bool = True
    skip_prompt: bool = False
    max_new_tokens: int = 200
    fp16: bool = True
    with_cost_aware_loss: bool = False
    alpha: float = 0.0,
    save_dataset_dir: Optional[str] = None
    save_model_pretrain_dir: Optional[str] = None
    deepspeed: Optional[str] = None


    

SAVED_ARGS = {
    "controller_warmup_fixed_input": TrainingArgs(
        prop_config=DynamicPropagationConfig(controller_layers=[1,2,3,4,5,6,7,8,9,10], with_fixed_input=True),
        batch_size=4,
        model='logs/opt-125m/databricks-dolly-15k_23-07_14-13-33/checkpoint-8000',
        train_epochs=3,
        eval_steps = 2000,
        save_strategy = EvaluationStrategy.NO,
        ddp=False,
        fp16=False,
        load_model_from_disk=True,
        deepspeed='ds_config.json',
        alpha = 3,
        with_cost_aware_loss=True,
        max_seq_length=256,
        gradient_checkpointing=True
    ),
    "controller_warmup": TrainingArgs(
        prop_config=DynamicPropagationConfig(controller_layers=[1,2,3,4,5,6,7,8,9,10]),
        batch_size=4,
        model='logs/opt-125m/databricks-dolly-15k_23-07_14-13-33/checkpoint-8000',
        train_epochs=3,
        eval_steps = 2000,
        save_strategy = EvaluationStrategy.NO,
        ddp=False,
        fp16=False,
        load_model_from_disk=True,
        deepspeed='ds_config.json',
        alpha = 3,
        with_cost_aware_loss=True,
        max_seq_length=256,
        gradient_checkpointing=True
    ),
    "full_prop_opt125_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='facebook/opt-125m',
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=256,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "full_prop_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='facebook/opt-350m',
        train_epochs=3,
        eval_steps = 200,
        save_strategy = "epoch",
        ddp=True,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=2,
        max_new_tokens=10,
        fp16 = False
    ),
    "full_prop_125_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='facebook/opt-125m',
        train_epochs=3,
        max_seq_length=256,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        ddp=True,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=2,
        max_new_tokens=100,
        fp16 = False
    ),
    "full_prop_350_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='facebook/opt-350m',
        train_epochs=3,
        max_seq_length=768,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        ddp=True,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=2,
        fp16 = False
    ),
    "full_prop_1.3_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='facebook/opt-1.3b',
        train_epochs=5,
        max_seq_length=768,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        ddp=True,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=2,
        fp16 = True
    ),
    "static_skip_args_load": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[2, 6, 8]),
        batch_size=5,
        train_epochs=3,
        load_dataset_from_disk=True,
        load_model_from_disk=True,
        dataset='dolly-15k',
        model="results/pre_train/opt-125M-static-skip",
        deepspeed='ds_config.json'
    ),
    "static_skip_args": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[2, 6, 8]),
        batch_size=10,
        train_epochs=3,
        fp16=False,
        ddp=False,
    ),
    "stochastic_dropout_args": TrainingArgs(
        prop_config=StochasticDropoutPropagationConfig(skip_probs=[0.0]*4 + [0.3]*8),
        batch_size=10,
        train_epochs=3,
    ),
    "eval_full_prop_125_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='results/opt-125m/databricks-dolly-15k_24-07_10-17-36/checkpoint-1501',
        load_model_from_disk = True,
        train_epochs=3,
        max_seq_length=256,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        ddp=False,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=2,
        max_new_tokens=100,
        fp16 = False
    ),
    "eval_full_prop_350_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='results/opt-350m/databricks-dolly-15k_23-07_21-08-20/checkpoint-564',
        load_model_from_disk = True,
        train_epochs=3,
        max_seq_length=768,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        ddp=False,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=2,
        fp16 = False
    ),
}

DATASET_KEYS ={
    "databricks/databricks-dolly-15k": {
        "prompt": "instruction",
        "context": "context",
        "response": "response"
    }
}
