'''
This file runs inference on the trained model to extract hidden states for each token in each sequence in the validation set.
'''
import os
from time import sleep
import datasets
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
from src.models.adalas_opt.modeling_adalas_opt import AdalasOPTForCausalLM
from src.utils.train_utils import DataCollatorForSeq2SeqGenerate
from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig
from src.utils.utils import get_abs_path, fix_the_seed, get_args
from transformers import AddedToken
from transformers import AutoTokenizer
from datetime import datetime




#set os parameters
os.environ["TOKENIZERS_PARALLELISM"] = "false" #parallel tokenizer causes issue in distributed mode

NUM_LAYERS = 24 #important to know how many layers the model has

def get_hidden_states(dataset,checkpoint,args,rank):
    
    #get hidden states:
    
    #get device
    device = torch.device("cuda:{}".format(rank))
    
    sep_token = AddedToken("<SEP>", lstrip=False, rstrip=False)
    tokenizer = AutoTokenizer.from_pretrained(
        get_abs_path([checkpoint]) if args.load_model_from_disk else checkpoint,
        padding_side='left', use_fast=False,
        sep_token=sep_token
    )

    sleep(rank*10)
    print(f"rank {rank} starting model loading")
    #load model and tokenizer
    adalas_config = AdalasOPTConfig.from_pretrained(get_abs_path([checkpoint]))
    #set config parameters
    adalas_config.propagation_config = args.prop_config
    adalas_config.with_cost_aware_loss = args.with_cost_aware_loss
    adalas_config.alpha = args.alpha
    adalas = AdalasOPTForCausalLM.from_pretrained(get_abs_path([checkpoint]),config=adalas_config).to(device)
    
    print("model loaded")
    
    batch_size = args.batch_size
    
    dataset = dataset.remove_columns(['instruction', 'input', 'output', 'text'])
    
    #get dataloader
    collator = DataCollatorForSeq2SeqGenerate(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=collator)
    
    #set model to eval mode
    adalas.eval()

    print("model in eval")
    
    num_layers = NUM_LAYERS
    num_tokens = 0
    #count number of label tokens
    for i, batch in enumerate(data_loader):
        batch_tokens = batch['labels'].view(-1).numpy()
        batch_tokens = np.where(batch_tokens != -100, batch_tokens, 1)
        batch_tokens = np.delete(batch_tokens,0)
        batch_tokens = np.append(batch_tokens,1) # 1 is the pad token
        label_tokens_in_batch = np.sum(batch_tokens != 1) 
        num_tokens += label_tokens_in_batch
        # tokens_per_batch = batch['input_ids'].size(1)
        # num_tokens += tokens_per_batch * batch['input_ids'].size(0)
    
   
    
    
    tensor_size_hidden = (num_layers+1,num_tokens,adalas_config.hidden_size) # +1 for input embeddings
    hidden_state_tensor = torch.empty(tensor_size_hidden,dtype=torch.float16,device='cpu')
    
    #we always need labels tensor
    labels_np = np.array([],dtype=np.int32)

    print("starting loop")
    
    #iterate through dataset
    torch_hidden_index = 0 #index to keep track of where to put hidden states in hidden_state_tensor
    for i, batch in enumerate(data_loader):
        #get input_ids
        input_ids = batch['input_ids'].to(device)
        #get attention_mask
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels']
        #flatten labels tensor along first dim
        batch_labels = batch_labels.view(-1)
        #process labels tensor
        #to numpy
        batch_labels_np = batch_labels.cpu().numpy()
        #replace -100 with pad token
        batch_labels_np= np.where(batch_labels_np != -100, batch_labels_np, 1) # 1 is the pad token
        #delete first label and add 1 eos token to the end
        batch_labels_np = np.delete(batch_labels_np,0)
        batch_labels_np = np.append(batch_labels_np,1) # 1 is the pad token
        mask = batch_labels_np != 1
        batch_labels_np = batch_labels_np[mask]
            
        #concatenate batch_labels to labels_np
        labels_np = np.concatenate((labels_np,batch_labels_np))
        
        
        with torch.no_grad():
            outputs = adalas(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )

    
        all_hidden_states = outputs.hidden_states
        #all_hidden_states is a tuple of length 25, each element is a tensor of shape (batch_size, num_tokens, hidden_size)
        
        #flatten along batch_size dim, loop through tuple and concatenate to hidden_state_tensor
        for j in range(len(all_hidden_states)):
            #flatten along batch_size dim
            #all_hidden_states[j] is now a tensor of shape (batch_size*num_tokens, hidden_size)
            #concatenate to hidden_state_tensor
            new_size = all_hidden_states[j].view(-1,all_hidden_states[j].size(2)).cpu().half()[mask,:].size(0)
            hidden_state_tensor[j,torch_hidden_index:torch_hidden_index+new_size,:] = all_hidden_states[j].view(-1,all_hidden_states[j].size(2)).cpu().half()[mask,:]
        torch_hidden_index += new_size
            

        if rank == 0 and i % 10 == 0:
            print(f"Completed batch {i}.")
        

    
    
    return labels_np, hidden_state_tensor
    
    
    
        
        
    
def run_inference(rank,world_size,dataset_dir,checkpoint,folderName,args):
    print(f"Running inference on rank {rank}.")
    #get validation data
    #load dataset
    dataset = datasets.load_from_disk(dataset_dir)
    dataset_val = dataset['test']
    #shard dataset into world_size parts, and only use the rank-th part
    dataset_val = dataset_val.shard(num_shards=world_size,index=rank,contiguous=True)
    
    print(f"Rank {rank} has {len(dataset_val)} samples.")
    #run inference
    labels_np,hidden_state_tensor = get_hidden_states(dataset_val,checkpoint,args,rank)
    
    
    #save to disk, convert to numpy
        
    np.save(get_abs_path([checkpoint]) + folderName + f'/labels_{rank}.npy',labels_np)
    
    
    hidden_state_tensor = hidden_state_tensor.numpy()
    np.save(get_abs_path([checkpoint]) + folderName + f'/hidden_state_tensor_{rank}.npy',hidden_state_tensor)
        
    
    print(f"Rank {rank} completed inference.")
       
    
def main():
    args = get_args()
    fix_the_seed(args.seed)
    world_size = 8
    #open JSON header file
    if args.tokenized_dataset_path is None:
        raise ValueError("Please provide a tokenized dataset path.")
    dataset_path = get_abs_path(['data','datasets',args.tokenized_dataset_path])
    model_path = args.model
    #get command line arguments using argparse

    
   
    
    now = datetime.now() # current date and time
    folderName = "/"+now.strftime("%m-%d-%Y_%H-%M")

    #make output directory for hidden state files
    os.makedirs(get_abs_path([model_path]) + folderName,exist_ok=True)
    
    #torch multiprocessing start method
    mp.spawn(run_inference, args=(world_size,dataset_path,model_path,folderName,args), nprocs=world_size, join=True)

    
    #post-join code
    print("Inference completed for all ranks. Aggregating results...")
        
    
    raw_final_hidden_state_np = None
    
    final_labels_np = None
    
    #aggregate results
    #read results from file, which are already numpy arrays
    for i in range(world_size):
        
        #load labels
        labels_np = np.load(get_abs_path([model_path]) + folderName + f'/labels_{i}.npy')
        if final_labels_np is None:
            final_labels_np = labels_np
        else:
            final_labels_np = np.concatenate((final_labels_np,labels_np),axis=0)
        
        hidden_state_np = np.load(get_abs_path([model_path]) + folderName + f'/hidden_state_tensor_{i}.npy')
        if raw_final_hidden_state_np is None:
            raw_final_hidden_state_np = hidden_state_np
        else:
            raw_final_hidden_state_np = np.concatenate((raw_final_hidden_state_np,hidden_state_np),axis=1)
        print(f"Rank {i} results aggregated.")
        
        
    
    
    #save labels to file
    np.save(get_abs_path([model_path]) + folderName + '/' + 'test_labels_np.npy',final_labels_np)
    
    
        
    #save results to file
    np.save(get_abs_path([model_path]) + folderName + '/' + 'test_hidden_state_np.npy',raw_final_hidden_state_np)
    
    #delete temporary files
    for i in range(world_size):
            
        os.remove(get_abs_path([model_path]) + folderName + f'/labels_{i}.npy')
       
        os.remove(get_abs_path([model_path]) + folderName + f'/hidden_state_tensor_{i}.npy')    
        
    
    print(f"Results saved to {get_abs_path([model_path]) + folderName}.")
    
    
    


if __name__ == '__main__':
    main()
    