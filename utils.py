import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    '''
    Define DiceLoss from dice coefficient.
    '''
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

def iou_score(y_true, y_pred, smooth=1):
    '''
    Calculate IoU score to check accuracy of Model
    y_true: Ground Truth
    y_pred: Prediction from model 
    smooth: epsilon not to make -inf
    '''
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = torch.abs(y_true * y_pred).sum()
    union = y_true.sum()+y_pred.sum() - torch.abs(y_true * y_pred).sum()
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    '''
    Calculate Dice score to check accuracy of Model
    y_true: Ground Truth
    y_pred: Prediction from model 
    smooth: epsilon not to make -inf
    '''
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = torch.abs(y_true * y_pred).sum()
    union = y_true.sum()+y_pred.sum()
    dice = ((2. * intersection + smooth)/(union + smooth)).mean()
    return dice

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        #print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

