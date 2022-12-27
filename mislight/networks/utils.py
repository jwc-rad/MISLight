import argparse
import copy
import importlib
import inspect
import functools
import numpy as np
import os
import pkgutil
from typing import Optional, Sequence, Tuple, Union
import yaml

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from mislight.utils.find_class import recursive_find_python_class

def define_network(net_config, net_module=None):
    ## use specialized loaders, or else use general class
    if net_module == 'mmseg':
        from mmcv.utils import Config
        from mmseg.models import build_segmentor
        import mmcv
        
        cfg = Config.fromfile(net_config)        
        net = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        net.init_weights()        
        return net
    
    if net_module is None:
        net_module = 'mislight.networks'
    
    with open(net_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    net_name = cfg['net_name']
    config = cfg['config']
    
    # parse args/opt if any
    for k in cfg.keys():
        if k in ['args', 'opt']:
            dummy = argparse.ArgumentParser()
            dummyopt = dummy.parse_args(args=[])
            dummyopt.__dict__.update(cfg[k])
            config[k] = dummyopt

    netcls = recursive_find_python_class(net_name, net_module)
                  
    if netcls:
        net = netcls(**config)       
        return net
    else:
        print(f'In {net_module}, there should be a class that matches {net_name}.')  
    return None


def load_pretrained_net(net, path):
    '''allow partial loading
    '''

    device = next(net.parameters()).device

    # load from checkpoint or state_dict
    print(f'trying to load pretrained from {path}')
    try:
        state_dict = torch.load(path, map_location=device)['state_dict']
    except:
        state_dict = torch.load(path, map_location=device)

    all_okay = True

    new_weights = net.state_dict()

    # partial loading. check key and shape
    for k in new_weights.keys():
        if not k in state_dict.keys():
            print(f'{k} is missing in pretrained')
            all_okay = False
        else:
            if new_weights[k].shape != state_dict[k].shape:
                print(f'skip {k}, required shape: {new_weights[k].shape}, pretrained shape: {state_dict[k].shape}')
                all_okay = False
            else:       
                new_weights[k] = state_dict[k]
    
    try:
        net.load_state_dict(new_weights)
        if all_okay:
            print('<All weights loaded successfully>')
    except:
        print(f'cannot load {path}. using intial net.')
        pass
    
    return net


def init_weights(net, init_type='xavier'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                init.xavier_normal_(m.weight)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    net.apply(init_func)
    return net