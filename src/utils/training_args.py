
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, \
    StochasticDropoutPropagationConfig, PropagationConfig, PropagationMode

@dataclass
class TrainingArgs:
    load_dataset_from_disk: bool = False
    dataset: str = 'databricks/databricks-dolly-15k'
    model: str = 'facebook/opt-125M'
    prop_config: PropagationConfig = PropagationConfig()
    batch_size: int = 10
    train_epochs: int = 3
    max_seq_length: int = 256
    promp_seq_length: float = 0.7
    from_checkpoint: bool = False
    multiprocess: bool = True
    instruction_template: str = "### User:"
    response_template: str = "\n### Assistant:"
    save_dataset_dir: Optional[str] = None
    save_model_pretrain_dir: Optional[str] = None

    

SAVED_ARGS = {
    
    "full_prop_args": TrainingArgs(
        prop_config=PropagationConfig(),
        batch_size=10,
        train_epochs=3,
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
