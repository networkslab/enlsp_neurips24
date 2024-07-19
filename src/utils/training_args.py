
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, \
    StochasticDropoutPropagationConfig, PropagationConfig, PropagationMode

@dataclass
class TrainingArgs:
    load_dataset_from_disk: bool = False
    load_model_from_disk: bool = False
    dataset: str = 'databricks/databricks-dolly-15k'
    model: str = 'facebook/opt-125M'
    prop_config: PropagationConfig = PropagationConfig()
    batch_size: int = 10
    train_epochs: int = 3
    max_seq_length: int = 256
    logging_steps:int = 20
    eval_steps:int = 100
    prompt_seq_length: float = 0.7
    from_checkpoint: bool = False
    multiprocess: bool = True
    instruction_template: str = "### User:"
    response_template: str = "\n### Assistant:"
    ddp: bool = True
    skip_prompt: bool = False
    save_dataset_dir: Optional[str] = None
    save_model_pretrain_dir: Optional[str] = None
    deepspeed: Optional[str] = None


    

SAVED_ARGS = {
    
    "full_prop_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=10,
        train_epochs=3,
    ),
    "full_prop_args_server_1": TrainingArgs(
        model='facebook/opt-350M',
        dataset='databricks/databricks-dolly-15k',
        prop_config=PropagationConfig(),
        batch_size=16,
        train_epochs=3,
        save_model_pretrain_dir='opt-350M-full',
        save_dataset_dir='dolly-15k',
        max_seq_length=1024,
        ddp=True,
        logging_steps=20,
        eval_steps=200,
        deepspeed='ds_config.json'
    ),
    "full_prop_args_server_2": TrainingArgs(
        load_dataset_from_disk= True,
        load_model_from_disk= True,
        model='results/pre_train/opt-350M-full',
        dataset='dolly-15k',
        prop_config=PropagationConfig(),
        batch_size=2,
        train_epochs=3,
        max_seq_length=1024,
        ddp=False,
        logging_steps=20,
        eval_steps=200,
        #deepspeed='ds_config.json'
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
    ),
    "stochastic_dropout_args": TrainingArgs(
        prop_config=StochasticDropoutPropagationConfig(skip_probs=[0.0]*4 + [0.3]*8),
        batch_size=10,
        train_epochs=3,
    )
}

DATASET_KEYS ={
    "databricks/databricks-dolly-15k": {
        "prompt": "instruction",
        "context": "context",
        "response": "response"
    }
}
