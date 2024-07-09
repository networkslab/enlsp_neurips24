from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, StochasticDropoutPropagationConfig
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM
from src.utils.utils import get_abs_path

MODEL_NAME = 'facebook/opt-125M'
DATASET_NAME = 'philschmid/dolly-15k-oai-style'

dataset = load_dataset(DATASET_NAME, split=Split.TRAIN)
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset[Split.TRAIN]
val_dataset = split_dataset[Split.TEST]
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
propagation_config = StochasticDropoutPropagationConfig(skip_probs=[0.0, 0.0, 0.0, 0.0] + [0.3] * 8)
adalas_config = AdalasOPTConfig.from_pretrained(MODEL_NAME)
adalas_config.propagation_config = propagation_config
adalas = AdalasOPTForCausalLM.from_pretrained(MODEL_NAME, config=adalas_config)

stripped_model_name = MODEL_NAME.split('/')[-1]
stripped_dataset_name = DATASET_NAME.split('/')[-1]
output_dir_name = f'{stripped_model_name}/{stripped_dataset_name}'

sft_config = SFTConfig(packing=False, output_dir=get_abs_path(['logs', output_dir_name]), max_seq_length=256,
                       report_to=['tensorboard'], logging_steps=20, logging_dir=get_abs_path(['logs', output_dir_name]),
                       logging_first_step=True, eval_steps=500, evaluation_strategy='steps')
trainer = SFTTrainer(
    model=adalas,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()