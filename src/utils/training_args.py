
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
    "cnndm_full_prop_1.3_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=2,
        dataset="abisee/cnn_dailymail",
        tokenized_dataset_path="cnn_dailymail",
        save_dataset_dir="cnn_dailymail",
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


