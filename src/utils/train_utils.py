import numpy as np
from evaluate import load
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTDecoder
from src.utils.utils import get_abs_path
import json
from torch import nn
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from trl.trainer import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, TrainingArguments, TrainerState, TrainerControl
from dataclasses import dataclass
import inspect
from src.utils.training_args import DATASET_KEYS
import copy
import pandas as pd
    

def compute_metrics(eval_pred,tokenizer, save_rouge=False, samples_to_save = 50):
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
    pred_percentage_length = [(float)((len(predictions[i])-len(labels[i])))/len(labels[i]) for i in range(len(predictions))]# TODO remove empty labels
    result["pred_percentage_length"] = np.mean(pred_percentage_length)
    
    #log examples for debugging
    examples = {}
    for i in range(samples_to_save):
        examples["input_"+str(i)] = str(inputs[i])
        examples["label_"+str(i)] = str(labels[i])
        examples["prediction_"+str(i)] = str(predictions[i])
    
    with open(f'{get_abs_path(["logs"])}/examples.json', "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)

    if save_rouge:
        #save individual rouge scores and sequence length
        r = rouge_score.compute(predictions=predictions, references=labels,
                                use_stemmer=False,rouge_types=["rouge1", "rouge2", "rougeL"],
                                use_aggregator=False)
        label_lengths = [label.size - np.count_nonzero(label == tokenizer.pad_token_id) for label in label_ids]
        prediction_lengths = [prediction.size-np.count_nonzero(prediction == tokenizer.pad_token_id) for prediction in prediction_ids]
        prompt_lengths = [input_ids[i].size - np.count_nonzero(input_ids[i] == tokenizer.pad_token_id) - label_lengths[i] for i in range(len(input_ids))]
        #save to pandas dataframe
        df = pd.DataFrame(
            {
                "rouge1": r["rouge1"],
                "rouge2": r["rouge2"],
                "rougeL": r["rougeL"],
                "label_length": label_lengths,
                "prediction_length": prediction_lengths,
                "prompt_length": prompt_lengths
            })
        #save to csv
        df.to_csv(f'{get_abs_path(["logs"])}/rouge_scores.csv', index=False)
    return {k: round(v,4) for k,v in result.items()}


def tokenize_and_format_dataset(dataset, dataset_name, tokenizer, args, instruction_template_ids, response_template_ids):
    
    #Tokenize
    def tokenize_function(examples):
        prompts = examples[DATASET_KEYS[dataset_name]['prompt']]
        #check if 'context' is in the datassetKeys 
        if 'context' in DATASET_KEYS[dataset_name]:
            contexts = examples[DATASET_KEYS[dataset_name]['context']]
            #concatenate prompts and contexts efficiently using map
            prompts = list(map(lambda x,y: x + '\n' + y, prompts, contexts))
            
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
        if 'context' in DATASET_KEYS[dataset_name]:
            contexts = examples[DATASET_KEYS[dataset_name]['context']]
            #concatenate prompts and contexts efficiently using map
            prompts = list(map(lambda x,y: x + '\n' + y, prompts, contexts))
            
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
    tokenized_dataset_val = dataset['test'].map(tokenize_function_eval, batched=True)
    
    return tokenized_dataset_train, tokenized_dataset_val
  



@dataclass
class SFTConfigGenerate(SFTConfig):
    
    eval_with_generate: Optional[bool] = False
    max_new_tokens: Optional[int] = 10


class SFTTrainerGenerate(SFTTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        """
        Overrides the SFTrainer's prediction step to use the generate method of the model. Implemetation copied from Seq2SeqTrainer.

        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        
        if self.args.eval_with_generate == False:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        generation_inputs = inputs.copy()

        generation_inputs['input_ids'] = generation_inputs['input_ids_for_gen'] #set input ids without the response part
        generation_inputs['attention_mask'] = generation_inputs['attention_mask_for_gen'] #set attention mask (only look at prompt)
        for key in ['input_ids_for_gen','attention_mask_for_gen','labels_for_gen']: #delete unused values, since they cause an error if they are kept
            generation_inputs.pop(key)
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**generation_inputs, max_new_tokens=self.args.max_new_tokens)
       
        for k in range(generated_tokens.size(dim=0)):
            prompt_length = generation_inputs['input_ids'][k].size(dim=0)
            generated_tokens[k][:prompt_length] = 1
        
        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
        labels_for_gen = inputs['labels_for_gen']
        for k in ['input_ids_for_gen', 'labels_for_gen', 'attention_mask_for_gen']:
            inputs.pop(k)
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = labels_for_gen
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
        
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

            #Addition: Add 'input_ids_for_gen', 'labels_for_gen' and 'attention_mask_for_gen' to _signature_columns so that they are not removed
            self._signature_columns += ['input_ids_for_gen','labels_for_gen','attention_mask_for_gen']
            #this will have no effect of those dict entries are not present

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
        percentage_skip_per_controller_per_seq = self.model.metrics['percentage_skip']
        for layer_idx, skip_per_seq in enumerate(percentage_skip_per_controller_per_seq):
            if len(skip_per_seq) > 0:
                skip_per_seq_tensor = torch.cat(skip_per_seq)
                output_tensors = [skip_per_seq_tensor.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(output_tensors, skip_per_seq_tensor) # gather all tensors from all processes
                skip_per_seq_tensor_gathered = torch.cat(output_tensors, dim=0)
                avg_perc_skip = torch.mean(skip_per_seq_tensor_gathered).item()
                if state.is_world_process_zero:
                    self.tb_writer.add_scalar(f'perc_skip/{layer_idx}', avg_perc_skip, state.global_step) # only log on one process
        self.model.flush_metrics()


def get_tensorboard_training_layout(decoder: AdalasOPTDecoder):
    layout = {
        "Additional training metrics": {
            "perc_skip": ["Multiline", [f'perc_skip/{cont_layer}' for cont_layer in range(len(decoder.layers))]],
        },
    }
    return layout
