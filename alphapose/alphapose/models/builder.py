from torch import nn

import sys
sys.path.insert(1, '/content/drive/MyDrive/pose-estimation-research/alphapose/alphapose/utils')
sys.path.insert(1, '/content/drive/MyDrive/pose-estimation-research/alphapose/alphapose/models')
sys.path.insert(1, '/content/drive/MyDrive/pose-estimation-research/alphapose/alphapose/datasets')

from registry import Registry, build_from_cfg, retrieve_from_cfg


SPPE = Registry('sppe')
LOSS = Registry('loss')
DATASET = Registry('dataset')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_sppe(cfg, preset_cfg, **kwargs):
    exec(f'from {str.lower(cfg.TYPE)} import {cfg.TYPE}')
    #print(eval(cfg.TYPE))
    SPPE.register_module(eval(cfg.TYPE))
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, SPPE, default_args=default_args)


def build_loss(cfg):
    return build(cfg, LOSS)


def build_dataset(cfg, preset_cfg, **kwargs):
    exec(f'from {str.lower(cfg.TYPE)} import {cfg.TYPE}')
    #DATASET.register_module(eval(cfg.TYPE))
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, DATASET, default_args=default_args)


def retrieve_dataset(cfg):
    exec(f'from {str.lower(cfg.TYPE)} import {cfg.TYPE}')
    DATASET.register_module(eval(cfg.TYPE))
    return retrieve_from_cfg(cfg, DATASET)
