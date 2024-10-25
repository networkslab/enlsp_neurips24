import numpy as np
from evaluate import load
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTDecoder
from src.utils.utils import get_abs_path, free
import json
from torch import nn
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from trl.trainer import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, TrainingArguments, TrainerState, TrainerControl
from dataclasses import dataclass
import inspect
import copy
import pandas as pd
import zlib
from src.utils.prepare_dataset import prepare_samsum, prepare_reddit, prepare_cnndm, prepare_alpaca
import pickle
import os

DATASET_KEYS ={
    "Samsung/samsum": {
        "prompt": "dialogue",
        "response": "summary",
        "prepare_fnc": prepare_samsum
    },
    "abisee/cnn_dailymail": {
        "prompt": "article",
        "response": "highlights",
        "prepare_fnc": prepare_cnndm
    },
    "tatsu-lab/alpaca": {
        "prompt": "instruction",
        "context": "input",
        "response": "output",
        "prepare_fnc": prepare_alpaca
    }
}
    

def compute_metrics(eval_pred,tokenizer, save_rouge=False,
                    samples_to_save = 20,fname="no_time",
                    pickle_file_params=None, generation_metrics=None):
    """Computes ROUGE score for evaluation predictions

    Args:
        eval_pred (_type_): _description_
        tokenizer (_type_): _description_
    """
    prediction_ids, label_ids, input_ids = eval_pred
    
    #decode predictions
    prediction_ids = np.where(prediction_ids == -100, tokenizer.pad_token_id, prediction_ids) # replace -100 with padding token
    predictions = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
    
    #decode labels
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids) # replace -100 with padding token
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    #decode inputs
    input_ids = np.where(input_ids == -100, tokenizer.pad_token_id, input_ids) # replace -100 with padding token
    inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    result = {}
    
    #compute rouge
    rouge_score = load("rouge")
    r = rouge_score.compute(predictions=predictions, references=labels, 
                            use_stemmer=False,rouge_types=["rouge1", "rouge2", "rougeL"],
                            use_aggregator=True)
    result["rouge1"] = r["rouge1"]
    result["rouge2"] = r["rouge2"]
    result["rougeL"] = r["rougeL"]
    
    #log the average error in length of the generated text as a fraction of the length of the label
    pred_percentage_length  = []
    for i in range(len(predictions)):
        if len(labels[i]) > 0:
            pred_percentage_length.append((float)((len(predictions[i])-len(labels[i])))/len(labels[i]))
    result["pred_percentage_length"] = np.mean(pred_percentage_length)

    # Calculate the individual rouge scores if save_rouge is True or pickle_file_params is provided
    if save_rouge or pickle_file_params is not None:
        r_ind = rouge_score.compute(predictions=predictions, references=labels,
                                    use_stemmer=False,rouge_types=["rouge1", "rouge2", "rougeL"],
                                    use_aggregator=False)
        label_lengths = [label.size - np.count_nonzero(label == tokenizer.pad_token_id) for label in label_ids]
        prediction_lengths = [prediction.size-np.count_nonzero(prediction == tokenizer.pad_token_id) for prediction in prediction_ids]
        prompt_lengths = [input_ids[i].size - np.count_nonzero(input_ids[i] == tokenizer.pad_token_id) - label_lengths[i] for i in range(len(input_ids))]

   
    is_process_zero = (dist.get_rank() == 0) if (dist.is_available() and dist.is_initialized()) else True
    # If pickle file name is provided, save the predictions, labels, inputs, and rouge scores to a pickle file
    # Target folder: Results/test_runs
    # File name: pickle_file_params in the format (eval_run_start_time, shard_number, model_name, dataset_name)
    if pickle_file_params is not None:
        if generation_metrics is not None:
            if dist.is_available() and dist.is_initialized():
                    
                    # Retrieve all distributed skip_count and token_count elements as lists
                    skip_counts = [generation_metrics['skip_count'].clone() for _ in range(dist.get_world_size())]
                    token_counts = [generation_metrics['token_count'].clone() for _ in range(dist.get_world_size())]
                    dist.all_gather(skip_counts, generation_metrics['skip_count'])
                    dist.all_gather(token_counts, generation_metrics['token_count']) 
                    
                    # Merge the distributed elements
                    skip_count = torch.zeros_like(skip_counts[0])
                    token_count = torch.tensor(0).to(skip_count.device)
                    for i in range(dist.get_world_size()):
                        skip_count += skip_counts[i]
                        token_count += token_counts[i]

            else:
                skip_count = generation_metrics['skip_count'].clone()
                token_count = generation_metrics['token_count'].clone()

        eval_run_start_time, shard_number, model_name, dataset_name = pickle_file_params
        file_name = f'{model_name}_{dataset_name}_{eval_run_start_time}'
        pickle_file_path = get_abs_path(["results", "test_runs", file_name])
        
        # Create the folder with the timestamp if it doesn't already exist
        if not os.path.exists(pickle_file_path):
            os.makedirs(pickle_file_path, exist_ok=True)
        
        # Save the file if is_process_zero
        if is_process_zero:
            with open(f'{pickle_file_path}/shard_{shard_number}.pkl', 'wb') as f:
    
                gen_metrics_dict = {
                    "predictions": predictions,
                    "labels": labels,
                    "inputs": inputs,
                    "rouge_1_avg": r["rouge1"],
                    "rouge_2_avg": r["rouge2"],
                    "rouge_L_avg": r["rougeL"],
                    "rouge_1_ind": r_ind["rouge1"],
                    "rouge_2_ind": r_ind["rouge2"],
                    "rouge_L_ind": r_ind["rougeL"],
                    "label_length": label_lengths,
                    "prediction_length": prediction_lengths,
                    "prompt_length": prompt_lengths
                }
    
                # Only add the skip percentages if the generation metrics are provided
                if generation_metrics is not None:
                    gen_metrics_dict["layer_skip_percentages"] = list(np.array(free(skip_count) / free(token_count))) #pickle does not support np
    
                pickle.dump(gen_metrics_dict, f)
    
    #log examples for debugging
    examples = {}
    for i in range(samples_to_save):
        examples["input_"+str(i)] = str(inputs[i])
        examples["label_"+str(i)] = str(labels[i])
        examples["prediction_"+str(i)] = str(predictions[i])
    
    with open(f'{get_abs_path(["logs"])}/examples.json', "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)

    if save_rouge:
        #save individual rouge scores, sequence length and hash of the prompt
        #hash the prompt
        hashes = [zlib.adler32(input.encode('utf-8')) for input in inputs] 
        #save to pandas dataframe
        df = pd.DataFrame(
            {
                "hash": hashes,
                "rouge1": r_ind["rouge1"],
                "rouge2": r_ind["rouge2"],
                "rougeL": r_ind["rougeL"],
                "label_length": label_lengths,
                "prediction_length": prediction_lengths,
                "prompt_length": prompt_lengths
            })
        #save to csv only if is_process_zero
        if is_process_zero:
            df.to_csv(f'{get_abs_path(["logs"])}/hashed_rouge_scores_{fname}.csv', index=False)
    return {k: round(v,4) for k,v in result.items()}


def tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids, context_template=None):
    
    #Tokenize
    def tokenize_function(examples):
        prompts = examples[DATASET_KEYS[dataset_name]['prompt']]
        #check if 'context' is in the datassetKeys 
        if 'context' in DATASET_KEYS[dataset_name] and examples[DATASET_KEYS[dataset_name]['context']] is not None:
            contexts = examples[DATASET_KEYS[dataset_name]['context']]
            #concatenate prompts and contexts efficiently using map
            prompts = list(map(lambda p,c: p + (context_template + c if len(c.strip()) > 0 else ''), prompts, contexts))
            
        responses = examples[DATASET_KEYS[dataset_name]['response']]
        
        #calculate max length of prompt, rounding down
        prompt_total_budget = int(args.max_seq_length * args.prompt_seq_length)
        prompt_budget = prompt_total_budget - len([tokenizer.bos_token_id]) - len(instruction_template_ids) - len(response_template_ids)
        response_budget = args.max_seq_length - prompt_total_budget - len([tokenizer.eos_token_id]) 
        #tokenize input
        prompt_tokens = tokenizer(prompts, truncation=True, max_length=prompt_budget, add_special_tokens=False)
        response_tokens = tokenizer(responses, truncation=True, max_length=response_budget, add_special_tokens=False)
        
        model_inputs = copy.deepcopy(prompt_tokens)
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i] = [tokenizer.bos_token_id] + instruction_template_ids + model_inputs['input_ids'][i] + response_template_ids
            model_inputs['input_ids'][i].extend(response_tokens['input_ids'][i])
            model_inputs['input_ids'][i].append(tokenizer.eos_token_id)            
            model_inputs['attention_mask'][i] = [1]*len(model_inputs['input_ids'][i])
        
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])


        #mask prompt from labels so loss is not calculated on it
        for i in range(len(model_inputs["labels"])):
            prompt_length = len(prompt_tokens['input_ids'][i]) + len(instruction_template_ids) + len(response_template_ids) + 1 # 1 for bos token
            model_inputs["labels"][i][:prompt_length] = [-100]*prompt_length #-100 means ignore loss for that token
        return model_inputs

    def tokenize_function_eval(examples):
        #### SAME AS tokenize_function ####
        prompts = examples[DATASET_KEYS[dataset_name]['prompt']]
        #check if 'context' is in the datassetKeys 
        if 'context' in DATASET_KEYS[dataset_name] and examples[DATASET_KEYS[dataset_name]['context']] is not None:
            contexts = examples[DATASET_KEYS[dataset_name]['context']]
            #concatenate prompts and contexts efficiently using map
            prompts = list(map(lambda p,c: p + (context_template + c if len(c.strip()) > 0 else ''), prompts, contexts))
            
        responses = examples[DATASET_KEYS[dataset_name]['response']]
        
        #calculate max length of prompt, rounding down
        prompt_total_budget = int(args.max_seq_length * args.prompt_seq_length)
        prompt_budget = prompt_total_budget - len([tokenizer.bos_token_id]) - len(instruction_template_ids) - len(response_template_ids)
        response_budget = args.max_seq_length - prompt_total_budget - len([tokenizer.eos_token_id]) 
        #tokenize input
        prompt_tokens = tokenizer(prompts, truncation=True, max_length=prompt_budget, add_special_tokens=False)
        response_tokens = tokenizer(responses, truncation=True, max_length=response_budget, add_special_tokens=False)
        
        model_inputs = copy.deepcopy(prompt_tokens)
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i] = [tokenizer.bos_token_id] + instruction_template_ids + model_inputs['input_ids'][i] + response_template_ids
        ######## NEW ########
        model_inputs['input_ids_for_gen'] = copy.deepcopy(model_inputs['input_ids'])
        model_inputs['labels_for_gen'] = response_tokens['input_ids']
        model_inputs['attention_mask_for_gen'] = []
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i].extend(response_tokens['input_ids'][i])
            model_inputs['input_ids'][i].append(tokenizer.eos_token_id)
            model_inputs['attention_mask_for_gen'].append([1]*len(model_inputs['input_ids_for_gen'][i]))
            model_inputs['attention_mask'][i] = [1]*len(model_inputs['input_ids'][i])
            model_inputs['labels_for_gen'][i].append(tokenizer.eos_token_id)
        
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
        # mask prompt from labels so loss is not calculated on it
        for i in range(len(model_inputs["labels"])):
            prompt_length = len(prompt_tokens['input_ids'][i]) + len(instruction_template_ids) + len(response_template_ids) + 1 # 1 for bos token
            model_inputs["labels"][i][:prompt_length] = [-100]*prompt_length #-100 means ignore loss for that token
        return model_inputs
        
    tokenized_dataset_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_dataset_val = dataset['validation'].map(tokenize_function_eval, batched=True)
    tokenized_dataset_test = dataset['test'].map(tokenize_function_eval, batched=True)
    
    return tokenized_dataset_train, tokenized_dataset_val, tokenized_dataset_test
  





class DataCollatorForSeq2SeqGenerate(DataCollatorForSeq2Seq):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        #Addition: Need to pad input_ids_for_gen, labels_for_gen and attention_mask_for_gen in the same way as labels. Copying code for padding labels
        #Designed to not have any impact if the specified dict entries are not present. We therefore don't have to create a new class
        ##########input_ids_for_gen########
        input_ids_for_gen = [feature['input_ids_for_gen'] for feature in features] if "input_ids_for_gen" in features[0].keys() else None
        # We have to pad the input_ids_for_gen before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if input_ids_for_gen is not None:
            max_input_ids_for_gen_length = max(len(l) for l in input_ids_for_gen)
            if self.pad_to_multiple_of is not None:
                max_input_ids_for_gen_length = (
                    (max_input_ids_for_gen_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side

            #Pad attention_mask_for_gen, using the same padding amount as input_ids_for_gen
            attention_mask_for_gen = [feature['attention_mask_for_gen'] for feature in features] if "attention_mask_for_gen" in features[0].keys() else None
            if attention_mask_for_gen is not None:
                for feature in features:
                    remainder = [0] * (max_input_ids_for_gen_length - len(feature["attention_mask_for_gen"]))
                    if isinstance(feature["attention_mask_for_gen"], list):
                        feature["attention_mask_for_gen"] = (
                        feature["attention_mask_for_gen"] + remainder if padding_side == "right" else remainder + feature["attention_mask_for_gen"]
                    )
                    elif padding_side == "right":
                        feature["attention_mask_for_gen"] = np.concatenate([feature["attention_mask_for_gen"], remainder])
                    else:
                        feature["attention_mask_for_gen"] = np.concatenate([remainder, feature["attention_mask_for_gen"]])


            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_input_ids_for_gen_length - len(feature["input_ids_for_gen"])) #using self.tokenizer.pad_token_id to be consistent
                if isinstance(feature["input_ids_for_gen"], list):
                    feature["input_ids_for_gen"] = (
                        feature["input_ids_for_gen"] + remainder if padding_side == "right" else remainder + feature["input_ids_for_gen"]
                    )
                elif padding_side == "right":
                    feature["input_ids_for_gen"] = np.concatenate([feature["input_ids_for_gen"], remainder]).astype(np.int64)
                else:
                    feature["input_ids_for_gen"] = np.concatenate([remainder, feature["input_ids_for_gen"]]).astype(np.int64)
        #######################

        ##########labels_for_gen########
        labels_for_gen = [feature["labels_for_gen"] for feature in features] if "labels_for_gen" in features[0].keys() else None
        # We have to pad the labels_for_gen before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels_for_gen is not None:
            max_labels_for_gen_length = max(len(l) for l in labels_for_gen)
            if self.pad_to_multiple_of is not None:
                max_labels_for_gen_length = (
                    (max_labels_for_gen_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_labels_for_gen_length - len(feature["labels_for_gen"]))
                if isinstance(feature["labels_for_gen"], list):
                    feature["labels_for_gen"] = (
                        feature["labels_for_gen"] + remainder if padding_side == "right" else remainder + feature["labels_for_gen"]
                    )
                elif padding_side == "right":
                    feature["labels_for_gen"] = np.concatenate([feature["labels_for_gen"], remainder]).astype(np.int64)
                else:
                    feature["labels_for_gen"] = np.concatenate([remainder, feature["labels_for_gen"]]).astype(np.int64)

        ##############################
        #End of Addition

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
    
    
class MetricsCallback(TensorBoardCallback):
    def __init__(self, summary_writer: SummaryWriter, model: AdalasOPTDecoder):
        super().__init__(summary_writer)
        self.model = model


    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        phase = 'train' if self.model.training else 'eval' # move this to an enum
        percentage_skip_per_controller_per_seq = self.model.metrics[phase]['percentage_skip']
        cross_layer_avg_perc_skip = []
        for layer_idx, skip_per_seq in enumerate(percentage_skip_per_controller_per_seq):
            if (skip_per_seq is not None) and len(skip_per_seq) > 0:
                #check if process is distributed
                if dist.is_available() and dist.is_initialized():
                    output_tensors = [skip_per_seq.clone() for _ in range(dist.get_world_size())]
                    dist.all_gather(output_tensors, skip_per_seq) # gather all tensors from all processes
                    skip_per_seq_tensor_gathered = torch.cat(output_tensors, dim=0)
                    avg_perc_skip = torch.mean(skip_per_seq_tensor_gathered).item()
                    if state.is_world_process_zero:
                        cross_layer_avg_perc_skip.append(avg_perc_skip)
                        self.tb_writer.add_scalar(f'perc_skip_{phase}/{layer_idx}', avg_perc_skip, state.global_step) # only log on one process
                else:
                    avg_perc_skip = torch.mean(skip_per_seq).item()
                    cross_layer_avg_perc_skip.append(avg_perc_skip)
                    self.tb_writer.add_scalar(f'perc_skip_{phase}/{layer_idx}', avg_perc_skip, state.global_step)
        if dist.is_available() and dist.is_initialized() and state.is_world_process_zero:
            self.tb_writer.add_scalar(f'perc_skip_{phase}/avg', np.mean(cross_layer_avg_perc_skip), state.global_step)
        elif not dist.is_available():
            self.tb_writer.add_scalar(f'perc_skip_{phase}/avg', np.mean(cross_layer_avg_perc_skip), state.global_step)
        self.model.flush_train_val_metrics(phase)

    # Access the model's generation metrics and reset them at the end of every evaluation phase
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.model._init_generation_metrics()

def get_tensorboard_training_layout(decoder: AdalasOPTDecoder):
    layout = {
        "Additional training metrics": {
            "perc_skip_train": ["Multiline", [f'perc_skip_train/{cont_layer}' for cont_layer in range(len(decoder.layers))]],
        },
        "Additional validation metrics": {
            "perc_skip_eval": ["Multiline", [f'perc_skip_eval/{cont_layer}' for cont_layer in range(len(decoder.layers))]],
        },
    }
    return layout
