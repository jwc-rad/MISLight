import re
import torch.nn as nn
from .utils import softmax_helper
from .dice_loss import DiceLoss
from .crossentropy import CrossEntropyLoss, KLDivLoss
from .error_loss import MSELoss
from .compound_loss import DiceCELoss, DiceFocalLoss

def define_loss(loss_name):
    ''' all use "forward(X, Y)"
    X should be logits (b,c,x,y,(z,))   
    Y should label/one-hot for label losses, else should be logits (b,c,x,y,(z,))
    '''
    loss = None
    # supervised losses
    if loss_name.lower() == 'label_dice':
        dc_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False)
        gen_kwargs = dict(apply_nonlin=[True, False], pre_avg=True, argmax=True)
        loss = DiceLoss(dc_kwargs, **gen_kwargs)
    elif loss_name.lower() == 'label_ce':
        ce_kwargs = dict()
        gen_kwargs = dict(apply_nonlin=[False, False], pre_avg=True, argmax=True)
        loss = CrossEntropyLoss(ce_kwargs, *gen_kwargs)
    elif loss_name.lower() == 'label_dc_ce':
        dc_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False)
        dc_gen_kwargs = dict(apply_nonlin=[True, False], pre_avg=True, argmax=True)
        ce_kwargs = dict()
        ce_gen_kwargs = dict(apply_nonlin=[False, False], pre_avg=True, argmax=True)
        loss = DiceCELoss(dc_kwargs, dc_gen_kwargs, ce_kwargs, ce_gen_kwargs, weight_dice=1, weight_ce=1)
    elif loss_name.lower() == 'label_dc_focal':
        dc_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False)
        dc_gen_kwargs = dict(apply_nonlin=[True, False], pre_avg=True, argmax=True)
        ce_kwargs = dict(alpha=0.5, gamma=2, smooth=1e-5)
        ce_gen_kwargs = dict(apply_nonlin=[True, False], pre_avg=True, argmax=True)
        loss = DiceFocalLoss(dc_kwargs, dc_gen_kwargs, ce_kwargs, ce_gen_kwargs, weight_dice=1, weight_ce=1)

    # unsupervised losses
    elif loss_name.lower() == 'hard_dice':
        dc_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False)
        generic_kwargs = dict(apply_nonlin=[True, True], pre_avg=True, argmax=True)
        loss = DiceLoss(dc_kwargs, **generic_kwargs)
    elif loss_name.lower() == 'hard_dc_ce':
        dc_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False)
        dc_gen_kwargs = dict(apply_nonlin=[True, True], pre_avg=True, argmax=True)
        ce_kwargs = dict()
        ce_gen_kwargs = dict(apply_nonlin=[False, True], pre_avg=True, argmax=True)
        loss = DiceCELoss(dc_kwargs, dc_gen_kwargs, ce_kwargs, ce_gen_kwargs, weight_dice=1, weight_ce=1)
    elif loss_name.lower() == 'hard_dc_focal':
        dc_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False)
        dc_gen_kwargs = dict(apply_nonlin=[True, True], pre_avg=True, argmax=True)
        ce_kwargs = dict()
        ce_kwargs = dict(alpha=0.5, gamma=2, smooth=1e-5)
        ce_gen_kwargs = dict(apply_nonlin=[True, True], pre_avg=True, argmax=True)
        loss = DiceFocalLoss(dc_kwargs, dc_gen_kwargs, ce_kwargs, ce_gen_kwargs, weight_dice=1, weight_ce=1)
        
    elif loss_name.lower() == 'kldiv':
        generic_kwargs = dict(apply_nonlin=[True, True], pre_avg=True)
        loss = KLDivLoss(**generic_kwargs)           
    elif loss_name.lower() == 'softmaxmse':
        generic_kwargs = dict(apply_nonlin=[True, True], pre_avg=True)
        loss = MSELoss(**generic_kwargs)
    elif loss_name.lower() == 'simplemse':
        generic_kwargs = dict(apply_nonlin=[False, False], pre_avg=False)
        loss = MSELoss(**generic_kwargs)
         
    else:
        raise NotImplementedError(f'loss name [{loss_name}] is not recognized')
        
    return loss


def check_loss(loss_name, temperature):
    temp_x, temp_y = temperature
    if loss_name.lower() in ['kldiv', 'simplemse']:
        assert temp_x == temp_y
        compensation = temp_x * temp_y
    else:
        compensation = 1
    return 
