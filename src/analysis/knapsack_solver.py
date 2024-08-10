from tqdm import tqdm
import numpy as np


class KnapsackSolver:
    def __init__(self,max_cap: int, values_ds: np.array, weights_ds: np.array):
        self.max_cap = max_cap
        self.values_ds = values_ds
        self.weights_ds = weights_ds
        self.num_groups = self.values_ds.shape[0]
        self.dp_mat = None
        self.weight_mat = None
    
    def reduction(values: list[float], weights: list[float]):
        '''Removes all entries where a lower value is associated with a higher weight'''
        pass

    def solve(self):
        """:param max_cap: maximum capacity over the entire dataset (the knapsack capacity). Should be an int for
        instance 3000 x 24
        values (3000 x 4) are assumed to be sorted in decreasing order across the columns
        """

        self.dp_mat = np.zeros((self.num_groups, self.max_cap + 1),dtype=np.float32) - np.inf
        self.weight_mat = np.zeros((self.num_groups, self.max_cap + 1),dtype=int) #keep track of the weights of the items that were chosen
        cum_min_weight = 0
        # handle first group
        g = 0
        values = self.values_ds[g]
        weights = self.weights_ds[g]
        min_weight = np.min(weights)
        cum_min_weight += min_weight
        for current_weight in range(min_weight, self.max_cap + 1):
            for idx in range(len(weights)):
                if weights[idx] <= self.max_cap:
                    self.dp_mat[0][current_weight] = values[idx]
                    self.weight_mat[0][current_weight] = weights[idx]
                    break

        for g in tqdm(range(1, len(self.values_ds))): # handle remaining groups
            values = self.values_ds[g]
            weights = self.weights_ds[g]
            min_weight = np.min(weights)
            cum_min_weight += min_weight
            for current_weight in range(cum_min_weight, self.max_cap + 1):
                best_value_in_group = -1 * np.inf
                best_choice_weight = None
                for idx in range(len(values)):
                    current_value = values[idx] + self.dp_mat[g - 1][current_weight - weights[idx]]
                    if current_value > best_value_in_group:
                        best_value_in_group = current_value
                        best_choice_weight = weights[idx]
                self.dp_mat[g][current_weight] = best_value_in_group
                self.weight_mat[g][current_weight] = best_choice_weight
    

        return self.dp_mat[-1]
    
    
    def reconstruct_chosen_items(self,chosen_cap: int):
        #reconstruct the items chosen as a list
        chosen_items = []
        current_weight = chosen_cap
        for g in range(self.num_groups - 1, -1, -1):
            chosen_items.insert(0, self.weight_mat[g][current_weight])
            current_weight -= self.weight_mat[g][current_weight]
        
        return chosen_items







