
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
    dataset: str = 'Samsung/samsum'
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
    load_best_model_at_end: bool = False
    save_total_limit: int = 20
    multiprocess: bool = True
    instruction_template: str = "### User:" #deprecated
    response_template: str = "\n### Assistant:" #deprecated
    context_template: Optional[str] = None
    ddp: bool = True
    skip_prompt: bool = False
    max_new_tokens: int = 200
    generate_do_sample: bool = False
    generate_temperature: float = 1.0
    fp16: bool = False
    with_cost_aware_loss: bool = False
    alpha: float = 0.0
    save_dataset_dir: Optional[str] = None
    save_model_pretrain_dir: Optional[str] = None
    deepspeed: Optional[str] = None
    with_lora: bool = False
    lora_rank: Optional[int] = 8
    lora_alpha: Optional[int] = 8
    lora_dropout: Optional[float] = 0.05
    testing_mode: bool = False
    num_test_shards: int = 5


SAVED_ARGS = {
    "alpaca_full_prop_1.3_args": TrainingArgs(
        learning_rate=2e-5,
        prop_config=PropagationConfig(),
        batch_size=8,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="alpaca",
        model='facebook/opt-iml-1.3b',
        train_epochs=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_new_tokens=300,
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "alpaca_ULS_12L_1.3_args": TrainingArgs(
        learning_rate=2e-5,
        prop_config=StaticSkipPropagationConfig(skip_layers=[1,3,5,7,9,11,13,15,17,19,21,23],freeze_skipped=True),
        batch_size=8,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="alpaca",
        load_model_from_disk=True,
        model='INSERT_CHECKPOINT_PATH',
        train_epochs=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_new_tokens=300,
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "alpaca_HS_1.3_args": TrainingArgs(
        learning_rate=2e-5,
        prop_config=DynamicPropagationConfig(controller_layers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]),
        batch_size=8,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="alpaca",
        load_model_from_disk=True,
        model='INSERT_CHECKPOINT_PATH',
        train_epochs=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_new_tokens=300,
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        max_seq_length=512,
        prompt_seq_length=0.25,
        with_cost_aware_loss=True,
        gradient_checkpointing=True,
        ddp=True,
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "alpaca_fixed_1.3_args": TrainingArgs(
        learning_rate=2e-5,
        prop_config=DynamicPropagationConfig(controller_layers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],with_fixed_input = True),
        batch_size=8,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="alpaca",
        load_model_from_disk=True,
        model='INSERT_CHECKPOINT_PATH',
        train_epochs=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_new_tokens=300,
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        max_seq_length=512,
        prompt_seq_length=0.25,
        with_cost_aware_loss=True,
        gradient_checkpointing=True,
        ddp=True,
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "eval_alpaca_full_prop_1.3_args": TrainingArgs(
        learning_rate=2e-5,
        prop_config=PropagationConfig(),
        batch_size=8,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="alpaca",
        load_model_from_disk=True,
        model='INSERT_CHECKPOINT_PATH',
        train_epochs=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_new_tokens=300,
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        deepspeed=None,
        fp16 = False
    ),
    "cnndm_full_prop_1.3_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=2,
        dataset="abisee/cnn_dailymail",
        save_dataset_dir="cnn_dailymail",
        model='facebook/opt-iml-1.3b',
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


