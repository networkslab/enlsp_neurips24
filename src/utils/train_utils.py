import numpy as np
from evaluate import load
from src.utils.utils import get_abs_path
import json
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import torch
from trl.trainer import SFTTrainer

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
    r = rouge_score.compute(predictions=predictions, references=labels, use_stemmer=False,rouge_types=["rouge1", "rouge2", "rougeL"],use_agregator=True)
    result["rouge1"] = r["rouge1"]
    result["rouge2"] = r["rouge2"]
    result["rougeL"] = r["rougeL"]
    
    #log the average error in length of the generated text as a fraction of the length of the label
    pred_percentage_length = [(float)((len(predictions[i])-len(labels[i])))/len(labels[i]) for i in range(len(predictions))]
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


def formatting_function_dolly_15k_oai_style(example, instruction_template, response_template):
    """Formats the input data for the dataset: "philschmid/dolly-15k-oai-style" 
    """
    output_texts = []
    for i in range(len(example["messages"])):
        text = f"{instruction_template}{example['messages'][0]['content']}\n{response_template}{example['messages'][1]['content']}"
        output_texts.append(text)
    return output_texts


class SFTTrainer_Generate(SFTTrainer):
    
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
        
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        generation_inputs = inputs.copy()
        
        #Addition: Modify generation inputs to use 'input_ids_for_gen' and 'labels_for_gen'
        #check if correct dict entries exist, so that normal behavior is not affected
        custom_generation = False #keep track of if we are modifying the generate behavior
        if('input_ids_for_gen' in generation_inputs.keys() and 'attention_mask_for_gen' in generation_inputs.keys() and 'labels_for_gen' in generation_inputs.keys()):
            custom_generation = True
            generation_inputs['input_ids'] = generation_inputs['input_ids_for_gen'] #set input ids
            generation_inputs['attention_mask'] = generation_inputs['attention_mask_for_gen'] #set attention mask
            for key in ['input_ids_for_gen','attention_mask_for_gen','labels_for_gen']: #delete unused values, since they cause an error if they are kept
                generation_inputs.pop(key)
        #end of addition

        
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**generation_inputs)
        
        #Addition: Remove prompt from generated tokens
        if custom_generation:
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
        
        