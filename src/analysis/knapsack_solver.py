from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import math


class KnapsackSolver:
    def __init__(self, values_ds: np.array, weights_ds: np.array, indices_ds: np.array, shard_number: int=1, max_layer: int=24):
        self.shard_size = shard_number # process will be divided into this many shards
        #split values and weights into shards
        self.values_ds = np.array_split(values_ds, shard_number)
        self.weights_ds = np.array_split(weights_ds, shard_number)
        self.indices_ds = np.array_split(indices_ds, shard_number)
        self.num_groups = [self.values_ds[i].shape[0] for i in range(shard_number)]
        self.max_caps = [np.sum(np.max(self.weights_ds[i], axis=1)) for i in range(shard_number)]
        self.dp_mat = [None] * shard_number
        self.indices_mat = [None] * shard_number
        self.total_groups = float(sum(self.num_groups))
        self.total_max_cap = sum(self.max_caps)
        self.max_layer = max_layer
        
    
    def reduction(values: list[float], weights: list[float]):
        '''Removes all entries where a lower value is associated with a higher weight'''
        pass
    
    def solve(self):
        '''Solves the knapsack problem for the given dataset, using multiprocessing'''
        #start multiprocessing
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for i in range(self.shard_size):
            p = mp.Process(target=self.shard_solve, args=(i,return_dict))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        #store the results
        for i in range(self.shard_size):
            self.dp_mat[i], self.indices_mat[i] = return_dict[i]
            
            
        
        
    
    def get_optimal_value(self, beta: float):
        #beta is the average layer use
        cap = int(beta/self.max_layer * self.total_max_cap)
        #split cap according to num_groups in each shard
        caps = [self.max_caps[i] / self.total_max_cap * cap for i in range(self.shard_size)]
        values = [self.dp_mat[i][-1][int(caps[i])] for i in range(self.shard_size)]
        #weighted sum of values, NOT a weighted average
        values_weighted_sum = sum([values[i] * self.max_caps[i] / self.total_max_cap for i in range(self.shard_size)]) * self.shard_size # multiply by shard_size to get the total value
        rouge_score = values_weighted_sum / self.total_groups
        return rouge_score
        

    def shard_solve(self, s_rank, return_dict):

        dp_mat = np.zeros((self.num_groups[s_rank], self.max_caps[s_rank] + 1),dtype=np.float32) - np.inf
        indices_mat = np.zeros((self.num_groups[s_rank], self.max_caps[s_rank] + 1),dtype=int) #keep track of the weights of the items that were chosen
        cum_min_weight = 0
        # handle first group
        g = 0
        values = self.values_ds[s_rank][g]
        weights = self.weights_ds[s_rank][g]
        indices = self.indices_ds[s_rank][g]
        min_weight = np.min(weights)
        cum_min_weight += min_weight
        for current_weight in range(min_weight, self.max_caps[s_rank] + 1):
            for idx in range(len(weights)):
                if weights[idx] <= self.max_caps[s_rank]:
                    dp_mat[0][current_weight] = values[idx]
                    indices_mat[0][current_weight] = indices[idx]
                    break

        for g in tqdm(range(1, len(self.values_ds[s_rank]))): # handle remaining groups
            values = self.values_ds[s_rank][g]
            weights = self.weights_ds[s_rank][g]
            indices = self.indices_ds[s_rank][g]
            min_weight = np.min(weights)
            cum_min_weight += min_weight
            for current_weight in range(cum_min_weight, self.max_caps[s_rank] + 1):
                best_value_in_group = -1 * np.inf
                best_choice_weight = None
                for idx in range(len(values)):
                    dp_mat_idx = current_weight - weights[idx] 
                    dp_mat_idx = max(dp_mat_idx, 0)
                    current_value = values[idx] + dp_mat[g - 1][dp_mat_idx]
                    if current_value > best_value_in_group:
                        best_value_in_group = current_value
                        best_choice_weight = indices[idx]
                dp_mat[g][current_weight] = best_value_in_group
                indices_mat[g][current_weight] = best_choice_weight
        
        return_dict[s_rank] = (dp_mat, indices_mat)
    
    
    def reconstruct_chosen_items(self, beta: float):
        #reconstruct the items chosen as a list
        #split cap according to num_groups in each shard
        caps = [self.max_caps[i] * beta/self.max_layer for i in range(self.shard_size)]
        chosen_items = []
        for i in range(self.shard_size):
            #round down to the nearest integer
            current_weight = math.floor(caps[i])
            for g in range(self.num_groups[i] - 1, -1, -1):
                chosen_item = self.indices_mat[i][g][current_weight]
                chosen_items.insert(0, chosen_item)
                idx = self.indices_ds[i][g].tolist().index(chosen_item)
                current_weight -= self.weights_ds[i][g][idx]
        return chosen_items
    







