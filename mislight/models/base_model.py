import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):       
    def __init__(self, opt):
        super().__init__()
        
        self.inference = opt.inference
        self.gpu_ids = opt.gpu_ids 
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def get_current_results(self):
        '''Return results. Trainer's Callback handles this.
        '''
        visual_ret = OrderedDict()
        for name in self.result_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
                    
    def load_pretrained(self, path):
        self.load_pretrained_nets(path, nets=self.model_names)
                    
    def load_pretrained_nets(self, path, nets=[]):
        '''For loading state_dict of part of the model.
        Loading full model should be done by "load_from_checkpoint" (Lightning)
        '''
        
        device = next(self.parameters()).device
        
        # load from checkpoint or state_dict
        print(f'trying to load pretrained from {path}')
        try:
            state_dict = torch.load(path, map_location=device)['state_dict']
        except:
            state_dict = torch.load(path, map_location=device)
        
        if len(nets)==0:
            self.load_state_dict(state_dict)
        
        all_keys_match = True
        for name in nets:
            if hasattr(self, 'net' + name):
                net = getattr(self, 'net' + name)
                new_weights = net.state_dict()
                
                # first check if pretrained has all keys
                keys_match = True
                for k in new_weights.keys():
                    if not f'net{name}.{k}' in state_dict.keys():
                        keys_match = False
                        all_keys_match = False
                        print(f"not loading {name} because keys don't match")
                        break
                if keys_match:
                    for k in new_weights.keys():
                        new_weights[k] = state_dict[f'net{name}.{k}']
                    net.load_state_dict(new_weights)
                        
        if all_keys_match:
            print('<All keys matched successfully>')
