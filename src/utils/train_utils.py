import numpy as np
from evaluate import load
from src.utils.utils import get_abs_path
import json
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import torch
from trl.trainer import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM
from dataclasses import dataclass
import inspect
import warnings
from collections.abc import Mapping
from src.utils.training_args import DATASET_KEYS
import copy


def compute_metrics(eval_pred,tokenizer):
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
    for i in range(50):
        examples["input_"+str(i)] = str(inputs[i])
        examples["label_"+str(i)] = str(labels[i])
        examples["prediction_"+str(i)] = str(predictions[i])
    
    with open(get_abs_path(["logs","examples.json"]), "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)
    
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
        prompt_budget = prompt_total_budget - len(tokenizer.bos_token_id) - len(instruction_template_ids) - len(response_template_ids)
        response_budget = args.max_seq_length - prompt_total_budget - len(tokenizer.eos_token_id) 
        #tokenize input
        prompt_tokens = tokenizer(prompts, truncation=True, max_length=prompt_budget, add_special_tokens=False)
        response_tokens = tokenizer(responses, truncation=True, max_length=response_budget, add_special_tokens=False)
        
        model_inputs = copy.deepcopy(prompt_tokens)
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i] += tokenizer.bos_token_id + instruction_template_ids + model_inputs['input_ids'][i] + response_template_ids
            model_inputs['input_ids'][i].extend(response_tokens['input_ids'][i])
            model_inputs['input_ids'][i].append(tokenizer.eos_token_id)            
            model_inputs['attention_mask'][i] = [1]*len(model_inputs['input_ids'][i])
        
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])

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
        prompt_budget = prompt_total_budget - len(tokenizer.bos_token_id) - len(instruction_template_ids) - len(response_template_ids)
        response_budget = args.max_seq_length - prompt_total_budget - len(tokenizer.eos_token_id) 
        #tokenize input
        prompt_tokens = tokenizer(prompts, truncation=True, max_length=prompt_budget, add_special_tokens=False)
        response_tokens = tokenizer(responses, truncation=True, max_length=response_budget, add_special_tokens=False)
        
        model_inputs = copy.deepcopy(prompt_tokens)
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i] += tokenizer.bos_token_id + instruction_template_ids + model_inputs['input_ids'][i] + response_template_ids
        ######## NEW ########
        model_inputs['input_ids_for_gen'] = copy.deepcopy(model_inputs['input_ids'])
        model_inputs['attention_mask_for_gen'] = [1]*len(model_inputs['input_ids_for_gen'])
        model_inputs['labels_for_gen'] = response_tokens['input_ids']
        for i in range(len(model_inputs['input_ids'])):
            model_inputs['input_ids'][i].extend(response_tokens['input_ids'][i])
            model_inputs['input_ids'][i].append(tokenizer.eos_token_id)            
            model_inputs['attention_mask'][i] = [1]*len(model_inputs['input_ids'][i])
            model_inputs['labels_for_gen'][i].append(tokenizer.eos_token_id)
        
        model_inputs['labels'] = copy.deepcopy(model_inputs['input_ids'])
        
    tokenized_dataset_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_dataset_val = dataset['validation'].map(tokenize_function_eval, batched=True)
    
    return tokenized_dataset_train, tokenized_dataset_val
  



@dataclass
class SFTConfigGenerate(SFTConfig):
    
    eval_with_generate: Optional[bool] = False


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
        
        label_mask = generation_inputs['input_ids'] == generation_inputs['labels'] # 1 where response starts
        generation_inputs['input_ids'] = torch.logical_not(label_mask) * generation_inputs['input_ids']
        generation_inputs['attention_mask'] = torch.logical_not(label_mask) * generation_inputs['attention_mask']
        num_shifts = torch.sum(generation_inputs['input_ids'] == 0, 1)
        num_shifts = num_shifts.tolist()
        generation_inputs['input_ids'] = generation_inputs['input_ids'] + label_mask.int()
        for seq_idx, num_shift in enumerate(num_shifts):
            generation_inputs['input_ids'][seq_idx] = torch.roll(generation_inputs['input_ids'][seq_idx], num_shift)
            generation_inputs['attention_mask'][seq_idx] = torch.roll(generation_inputs['attention_mask'][seq_idx], num_shift)
        
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**generation_inputs, max_new_tokens=100) # TODO pass max_new_tokens as a config
       
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
            labels = inputs["labels"]
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

class DataCollatorForCompletionOnlyLMGenerate(DataCollatorForCompletionOnlyLM):
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        #Addition: pad 'labels','input_ids_for_gen', 'labels_for_gen' and 'attention_mask_for_gen'
        #TODO
        
        batch = super(DataCollatorForCompletionOnlyLM,self).torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch
