
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.trainer_utils import EvaluationStrategy
from src.models.controllers.controller_types import ControllerInputType

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, \
    StochasticDropoutPropagationConfig, PropagationConfig, PropagationMode, DynamicPropagationConfig, \
    StaticEEPropagationConfig


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
    save_total_limit: int = 20,
    multiprocess: bool = True
    instruction_template: str = "### User:" #deprecated
    response_template: str = "\n### Assistant:" #deprecated
    context_template: Optional[str] = None
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
    "alpaca_tiny_prop_350_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="opt350_alpaca_tiny",
        model='facebook/opt-350M',
        train_epochs=3,
        save_strategy="steps",
        gradient_accumulation_steps=5,
        save_steps=20,
        eval_steps=20,
        load_best_model_at_end=True,
        save_total_limit=2,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        tokenized_dataset_path="opt350_alpaca_tiny",
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "alpaca_opt350_gumbels_eval": TrainingArgs(
        prop_config=DynamicPropagationConfig(controller_layers=list(range(1, 23))),
        batch_size=4,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="opt350_alpaca",
        model='results/checkpoint-3640/alpaca_20-08_13-14-54/checkpoint-1820',
        train_epochs=1,
        load_model_from_disk=True,
        learning_rate=1e-5,
        save_strategy=EvaluationStrategy.STEPS,
        gradient_accumulation_steps=5,
        save_steps=910,
        eval_steps=910,
        load_best_model_at_end=True,
        save_total_limit=3,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        # tokenized_dataset_path="opt350_alpaca",
        deepspeed=None,
        fp16 = False
    ),
    "alpaca_opt350_gumbels_hs": TrainingArgs(
        prop_config=DynamicPropagationConfig(controller_layers=list(range(1, 23))),
        batch_size=4,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="opt350_alpaca",
        model='results/opt-350M/alpaca_19-08_18-52-58/checkpoint-3640',
        train_epochs=1,
        load_model_from_disk=True,
        learning_rate=1e-5,
        save_strategy=EvaluationStrategy.STEPS,
        gradient_accumulation_steps=5,
        save_steps=910,
        eval_steps=910,
        load_best_model_at_end=True,
        save_total_limit=3,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        # tokenized_dataset_path="opt350_alpaca",
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "alpaca_random_dropout_fine_tune_prop_350_args": TrainingArgs(
        prop_config=StochasticDropoutPropagationConfig(skip_probs=[0] + [0.35] * 22 + [0]),
        batch_size=1,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="opt350_alpaca",
        model='results/opt-350M/alpaca_18-08_11-52-14/checkpoint-5460',
        train_epochs=2,
        load_model_from_disk=True,
        learning_rate=1e-5,
        save_strategy="steps",
        gradient_accumulation_steps=5,
        save_steps=910,
        eval_steps=910,
        load_best_model_at_end=True,
        save_total_limit=3,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=False,
        tokenized_dataset_path="opt350_alpaca",
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "alpaca_full_prop_350_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=4,
        dataset="tatsu-lab/alpaca",
        save_dataset_dir="opt350_alpaca",
        model='facebook/opt-350M',
        train_epochs=3,
        save_strategy="steps",
        gradient_accumulation_steps=5,
        save_steps=910,
        eval_steps=910,
        load_best_model_at_end=True,
        save_total_limit=3,
        max_seq_length=512,
        prompt_seq_length=0.25,
        gradient_checkpointing=True,
        ddp=True,
        tokenized_dataset_path="opt350_alpaca",
        deepspeed='ds_config.json',
        fp16 = False
    ),
    "cnndm_full_prop_1.3_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=2,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail",
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


