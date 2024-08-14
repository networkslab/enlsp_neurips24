from tqdm import tqdm
import numpy as np
import multiprocessing as mp


class KnapsackSolver:
    def __init__(self, values_ds: np.array, weights_ds: np.array, shard_number: int=1):
        self.shard_size = shard_number # process will be divided into this many shards
        #split values and weights into shards
        self.values_ds = np.array_split(values_ds, shard_number)
        self.weights_ds = np.array_split(weights_ds, shard_number)
        self.num_groups = [self.values_ds[i].shape[0] for i in range(shard_number)]
        self.max_caps = [self.num_groups[i] * 24 for i in range(shard_number)]
        self.dp_mat = [None] * shard_number
        self.weight_mat = [None] * shard_number
        self.total_groups = float(sum(self.num_groups))
        
    
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
            self.dp_mat[i], self.weight_mat[i] = return_dict[i]
            
            
        
        
    
    def get_optimal_value(self, cap):
        #split cap according to num_groups in each shard
        caps = [self.num_groups[i] / self.total_groups * cap for i in range(self.shard_size)]
        values = [self.dp_mat[i][-1][int(caps[i])] for i in range(self.shard_size)]
        #weighted sum of values, NOT a weighted average
        values_weighted_sum = sum([values[i] * self.num_groups[i] / self.total_groups for i in range(self.shard_size)]) * self.shard_size # multiply by shard_size to get the total value
        return values_weighted_sum
        

    def shard_solve(self, s_rank, return_dict):
        """:param max_cap: maximum capacity over the entire dataset (the knapsack capacity). Should be an int for
        instance 3000 x 24
        values (3000 x 4) are assumed to be sorted in decreasing order across the columns
        """

        dp_mat = np.zeros((self.num_groups[s_rank], self.max_caps[s_rank] + 1),dtype=np.float32) - np.inf
        weight_mat = np.zeros((self.num_groups[s_rank], self.max_caps[s_rank] + 1),dtype=int) #keep track of the weights of the items that were chosen
        cum_min_weight = 0
        # handle first group
        g = 0
        values = self.values_ds[s_rank][g]
        weights = self.weights_ds[s_rank][g]
        min_weight = np.min(weights)
        cum_min_weight += min_weight
        for current_weight in range(min_weight, self.max_caps[s_rank] + 1):
            for idx in range(len(weights)):
                if weights[idx] <= self.max_caps[s_rank]:
                    dp_mat[0][current_weight] = values[idx]
                    weight_mat[0][current_weight] = weights[idx]
                    break

        for g in tqdm(range(1, len(self.values_ds[s_rank]))): # handle remaining groups
            values = self.values_ds[s_rank][g]
            weights = self.weights_ds[s_rank][g]
            min_weight = np.min(weights)
            cum_min_weight += min_weight
            for current_weight in range(cum_min_weight, self.max_caps[s_rank] + 1):
                best_value_in_group = -1 * np.inf
                best_choice_weight = None
                for idx in range(len(values)):
                    current_value = values[idx] + dp_mat[g - 1][current_weight - weights[idx]]
                    if current_value > best_value_in_group:
                        best_value_in_group = current_value
                        best_choice_weight = weights[idx]
                dp_mat[g][current_weight] = best_value_in_group
                weight_mat[g][current_weight] = best_choice_weight
        
        return_dict[s_rank] = (dp_mat, weight_mat)
    
    
    def reconstruct_chosen_items(self, chosen_cap: int):
        #reconstruct the items chosen as a list
        #split cap according to num_groups in each shard
        caps = [self.num_groups[i] / sum(self.num_groups) * chosen_cap for i in range(self.shard_size)]
        chosen_items = []
        for i in range(self.shard_size):
            current_weight = int(caps[i])
            for g in range(self.num_groups[i] - 1, -1, -1):
                chosen_items.insert(0, self.weight_mat[i][g][current_weight])
                current_weight -= self.weight_mat[i][g][current_weight]
        return chosen_items
    
    # def shard_reconstruct_chosen_items(self,chosen_cap: int):
    #     #reconstruct the items chosen as a list
    #     chosen_items = []
    #     current_weight = chosen_cap
    #     for g in range(self.num_groups - 1, -1, -1):
    #         chosen_items.insert(0, self.weight_mat[g][current_weight])
    #         current_weight -= self.weight_mat[g][current_weight]
        
    #     return chosen_items







