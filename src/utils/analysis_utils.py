import numpy as np
def cosine_similarity(layer_i_hidden_states, layer_j_hidden_states):
    dot_product = np.sum(layer_i_hidden_states * layer_j_hidden_states, axis=1)
    norm_product = np.linalg.norm(layer_i_hidden_states, axis=1) * np.linalg.norm(layer_j_hidden_states, axis=1)
    return dot_product / norm_product
