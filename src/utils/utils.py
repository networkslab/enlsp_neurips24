import argparse
import dataclasses
import os
from typing import Iterable

import numpy as np
import random
import torch

from src.models.adalas_opt.config_adalas_opt import PropagationMode
from src.utils.training_args import SAVED_ARGS, TrainingArgs

def fix_the_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("enlsp_neurips24")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def free(torch_tensor):
    return torch_tensor.cpu().detach().numpy()

def freeze_network(network, excluded_submodules: list[str], verbose = False):
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    # set everything to not trainable.
    for param in network.parameters():
        param.requires_grad = False

    for submodule_attr_name in excluded_submodules:  # Unfreeze excluded submodules to be trained.
        submodule = getattr(network, submodule_attr_name)
        if isinstance(submodule, Iterable):
            for submodule in getattr(network, submodule_attr_name): # iterate one level
                for param in submodule.parameters():
                    param.requires_grad = True
        else:
            for param in submodule.parameters():
                param.requires_grad = True

    if verbose:
        trainable_parameters = filter(lambda p: p.requires_grad,
                                      network.parameters())
        num_trainable_params = sum(
            [np.prod(p.size()) for p in trainable_parameters])
        print('Successfully froze network: from {} to {} trainable params.'.format(
            total_num_parameters, num_trainable_params))

def freeze_skipped_decoder_layers(network, excluded_submodules: list[str], skipped_layers, verbose = False):
    '''network should have an attribute called model, representing the backbone and this backbone should have a decoder attr.'''
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    freeze_network(network, excluded_submodules, False)

    for layer_idx, decoder_layer in enumerate(network.model.decoder.layers):
        if layer_idx not in skipped_layers:
            if verbose:
                print(f"Unfreezing decoder layer {layer_idx}")
            for param in decoder_layer.parameters():
                param.requires_grad = True
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  network.parameters())
    num_trainable_params = sum(
        [np.prod(p.size()) for p in trainable_parameters])
    print('Successfully froze network: from {} to {} trainable params.'.format(
        total_num_parameters, num_trainable_params))

def freeze_top_decoder_layers(network,
                              excluded_submodules: list[str],
                              last_unfrozen_layer, verbose = False):
    '''network should have an attribute called model, representing the backbone and this backbone should have a decoder attr.'''
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    freeze_network(network, excluded_submodules, False)

    for layer_idx, decoder_layer in enumerate(network.model.decoder.layers):
        if layer_idx <= last_unfrozen_layer:
            if verbose:
                print(f"Unfreezing decoder layer {layer_idx}")
            for param in decoder_layer.parameters():
                param.requires_grad = True
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  network.parameters())
    num_trainable_params = sum(
        [np.prod(p.size()) for p in trainable_parameters])
    print('Successfully froze network: from {} to {} trainable params.'.format(
        total_num_parameters, num_trainable_params))

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_args", type=str, default='full_prop_args')
    for field in dataclasses.fields(TrainingArgs): # introspection of fields to allow overriding any arg
        parser.add_argument(f"--{field.name}", type=field.type)

    parser_args = parser.parse_args()
    overwritten_args = {}
    for arg_name, value in parser_args.__dict__.items():
        if value is not None and arg_name != 'training_args':
            overwritten_args[arg_name] = value
    if parser_args.training_args not in SAVED_ARGS:
        raise ValueError(f"Training args {parser_args.training_args} not found in SAVED_ARGS")
    args = SAVED_ARGS[parser_args.training_args]
    args.update_fields(overwritten_args)
    validate_args(args)
    return args

def validate_args(args):
    if args.prop_config.propagation_mode == PropagationMode.STATIC_SKIP:
        assert len(args.prop_config.skip_layers) > 0, "STATIC SKIP needs a list of layers to skip"
    if args.prop_config.propagation_mode == PropagationMode.STOCHASTIC_DROPOUT:
        assert len(args.prop_config.skip_probs) > 0, 'STOCHASTIC DROPOUT needs a list of skip probabilities'


def free(torch_tensor):
    '''detaches a torch tensor, moves it to cpu and casts it to numpy array'''
    return torch_tensor.cpu().detach().numpy()