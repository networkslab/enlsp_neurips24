from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, IntervalStrategy
from datasets import load_dataset, Split
import argparse
from datetime import datetime

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, StaticSkipPropagationConfig, \
    StochasticDropoutPropagationConfig, PropagationConfig, PropagationMode
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM
from src.utils.utils import get_abs_path, list_of_ints, list_of_floats


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='philschmid/dolly-15k-oai-style')
    parser.add_argument("--arch", type=str, choices=['facebook/opt-125M', 'facebook/opt-250M'], default='facebook/opt-125M')
    parser.add_argument("--prop_mode", type=PropagationMode, choices=list(PropagationMode), default=PropagationMode.FULL)

    parser.add_argument("--skip_layers", help='Which layers to skip when using STATIC_SKIP propagation mode',
                        type=list_of_ints,default=[])
    parser.add_argument("--skip_probs", help='Probability of skipping each layer',
                        type=list_of_floats ,default=[])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--from_checkpoint",  action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_prompt",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiprocess", action='store_true')
    args = parser.parse_args()
    validate_args(args)

    MODEL_NAME = args.arch
    DATASET_NAME = args.dataset

    dataset = load_dataset(DATASET_NAME, split=Split.TRAIN)
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset[Split.TRAIN]
    val_dataset = split_dataset[Split.TEST]

    if args.prop_mode == PropagationMode.STATIC_SKIP:
        propagation_config = StaticSkipPropagationConfig(skip_layers=args.skip_layers)
    elif args.prop_mode == PropagationMode.STOCHASTIC_DROPOUT:
        propagation_config = StochasticDropoutPropagationConfig(skip_probs=args.skip_probs)
    else:
        propagation_config = PropagationConfig()
    adalas_config = AdalasOPTConfig.from_pretrained(MODEL_NAME)
    adalas_config.propagation_config = propagation_config
    adalas_config.skip_prompt = args.skip_prompt
    adalas = AdalasOPTForCausalLM.from_pretrained(MODEL_NAME, config=adalas_config)

    stripped_model_name = MODEL_NAME.split('/')[-1]
    stripped_dataset_name = DATASET_NAME.split('/')[-1]
    current_time_str = datetime.now().strftime("%d-%m_%H-%M-%S")

    output_dir_name = f'{stripped_model_name}/{stripped_dataset_name}_{current_time_str}'

    sft_config = SFTConfig(packing=False, output_dir=get_abs_path(['results', output_dir_name]), max_seq_length=256,
                           report_to=['tensorboard'], logging_steps=20, logging_dir=get_abs_path(['logs', output_dir_name]),
                           logging_first_step=True, eval_steps=500, evaluation_strategy='steps',
                           save_strategy=IntervalStrategy.NO)
    trainer = SFTTrainer(
        model=adalas,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

def validate_args(args):
    if args.prop_mode == PropagationMode.STATIC_SKIP:
        assert len(args.skip_layers) > 0, "STATIC SKIP needs a list of layers to skip"
    if args.prop_mode == PropagationMode.STOCHASTIC_DROPOUT:
        assert len(args.skip_probs) > 0, 'STOCHASTIC DROPOUT needs a list of skip probabilities'

if __name__ == "__main__":
    main()