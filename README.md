# Dynamic layer selection in decoder-only transformers
Code to reproduce experiments of the [NeurIPS ENLSP-IV workshop](https://neurips2024-enlsp.github.io/) paper "Dynamic Layer Selection in Decoder-only Transformers".

## Pre-requisites

* Install Python3.9
* Create a virtual environment (recommended)
* Activate your virtual environment
* Install dependencies by running: `pip install -r requirements.txt` from the main folder.

## Code overview

This repository allows to run the experiments reported in our paper. We used the OPT model throughout our experiments using HuggingFace.

We thus create a model that inherits from HuggingFace's OPT class and augment it several capabilities such as:
* KV cache propagation
* Support for Early-exit
* Support for layer skipping (in various modes such as static, stochastic and learnable). <br>
All these capabilities are added in the class AdalasOPT in `src/models/adalas_opt/modeling_adalas_opt.py`

## Running a script
There are several scripts in this repository:
1. train_model.py trains the AdalasOPT model in various possible modes.
2. warmup_controller allows to train the skip controllers while freezing the backbone (exp 2 in paper)
3. eval_model.py evaluates a trained model.

### Training arguments
The easiest way to pass training arguments is to define a TrainingArgument instance in training_args.py. The TrainingArgument class is a data class that encapsulates various arguments for training a model such as the model size, the dataset, the batch size, what skipping mechanism to use etc... The user can define a named instance of that class such as:
```
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
```
and then reference this set of args by simply running 
```
python run_train.py --training_args alpaca_full_prop_350_args
```

Here `alpaca_full_prop_350_args` is the name of the training arg instance.<br>
The user can also override single arguments of an instance of TrainingArgument:

```
python run_train.py --training_args alpaca_full_prop_350_args --train_epochs 5
```
To run in distributed mode:

```
 torchrun --standalone --nproc_per_node 8 run_train.py --training_args alpaca_full_prop_350_args
```
