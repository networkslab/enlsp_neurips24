from transformers import AutoModel, AutoTokenizer

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM

MODEL_NAME = 'facebook/opt-350M'
adalas_config = AdalasOPTConfig.from_pretrained(MODEL_NAME)
adalas = AdalasOPTForCausalLM.from_pretrained(MODEL_NAME, config=adalas_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# adalas.model.load_state_dict(pretrained_model.state_dict())
prompt = 'Today is a good day to'
input_ids = tokenizer.encode(prompt, return_tensors='pt')
generation = adalas.generate(input_ids=input_ids)
out = tokenizer.batch_decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)


