
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.trainer_utils import EvaluationStrategy
from src.models.controllers.controller_types import ControllerInputType

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, \
    StochasticDropoutPropagationConfig, PropagationConfig, PropagationMode, DynamicPropagationConfig, \
    StaticEEPropagationConfig, RandomForBudgetPropagationConfig


class DictOverwritable(object):
    '''allows to overwrite some attributes of a class with a dict'''
    def update_fields(self, dict_for_update):
        for key, value in dict_for_update.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class TrainingArgs(DictOverwritable):
    seed: int = 42
    learning_rate: float = 5e-5
    tokenized_dataset_path: str = None
    load_dataset_from_disk: bool = False #deprecated for tokenized_dataset_path
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
    save_total_limit: int = 3,
    multiprocess: bool = True
    instruction_template: str = "### User:"
    response_template: str = "\n### Assistant:"
    ddp: bool = True
    skip_prompt: bool = False
    max_new_tokens: int = 200
    generate_do_sample: bool = False
    generate_temperature: float = 1.0
    fp16: bool = True
    with_cost_aware_loss: bool = False
    alpha: float = 0.0
    save_dataset_dir: Optional[str] = None
    save_model_pretrain_dir: Optional[str] = None
    deepspeed: Optional[str] = None
    with_lora: bool = False
    lora_rank: Optional[int] = 8
    lora_alpha: Optional[int] = 8
    lora_dropout: Optional[float] = 0.05


SAVED_ARGS = {
     "hidden_state_opt125_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='facebook/opt-125m',
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="mini_cnn_dailymail",
        save_model_pretrain_dir="opt-125m",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=512,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "hidden_state_1.3_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        model='results/opt-iml-1.3b/cnn_dailymail_13-08_03-40-32/checkpoint-11500',
        load_model_from_disk = True,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail_1000_val",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=2038,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "hidden_state_1.3_LS20_args": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[1,7,13,19]),
        batch_size=4,
        model='results/opt-iml-1.3b/cnn_dailymail_13-08_03-40-32/checkpoint-11500',
        load_model_from_disk = True,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail_1000_val",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=2038,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "hidden_state_1.3_LS16_args": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[1,4,7,10,13,16,19,22]),
        batch_size=4,
        model='results/opt-iml-1.3b/cnn_dailymail_13-08_03-40-32/checkpoint-11500',
        load_model_from_disk = True,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail_1000_val",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=2038,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "hidden_state_1.3_LS12_args": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[1,3,5,7,9,11,13,15,17,19,21,23]),
        batch_size=4,
        model='results/opt-iml-1.3b/cnn_dailymail_13-08_03-40-32/checkpoint-11500',
        load_model_from_disk = True,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail_1000_val",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=2038,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "hidden_state_1.3_LS8_args": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23]),
        batch_size=4,
        model='results/opt-iml-1.3b/cnn_dailymail_13-08_03-40-32/checkpoint-11500',
        load_model_from_disk = True,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail_1000_val",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=2038,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "hidden_state_1.3_LS4_args": TrainingArgs(
        prop_config=StaticSkipPropagationConfig(skip_layers=[1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23]),
        batch_size=4,
        model='results/opt-iml-1.3b/cnn_dailymail_13-08_03-40-32/checkpoint-11500',
        load_model_from_disk = True,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail_1000_val",
        train_epochs=3,
        eval_steps = 1000,
        save_strategy = "steps",
        save_steps = 1000,
        fp16=False,
        gradient_checkpointing=False,
        ddp=False,
        max_seq_length=2038,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=1,
    ),
    "random_for_budget_test": TrainingArgs(
        prop_config=RandomForBudgetPropagationConfig(budget=9),
        batch_size=4,
        model='logs/opt-125m/databricks-dolly-15k_23-07_14-13-33/checkpoint-8000',
        train_epochs=3,
        eval_steps = 30,
        save_dataset_dir="dolly_opt125",
        save_strategy = EvaluationStrategy.NO,
        ddp=False,
        fp16=False,
        load_model_from_disk=True,
        deepspeed='ds_config.json',
        alpha = 10,
        with_cost_aware_loss=False,
        max_seq_length=256,
        tokenized_dataset_path='dolly_opt125',
        gradient_checkpointing=False,
        with_lora=False
    ),
    "cnndm_full_prop_1.3_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=2,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail",
        save_dataset_dir="cnn_dailymail_1000_val",
        model='facebook/opt-iml-1.3b',
        instruction_template= "### Article:",
        response_template= "\n### Summary:",
        train_epochs=2,
        save_strategy="steps",
        save_steps=1994,
        eval_steps=1994,
        max_seq_length=2048,
        ddp=True,
        deepspeed='ds_config.json',
        gradient_accumulation_steps=3,
        fp16 = False
    ),

}


