from __future__ import absolute_import
import torch
from torch import nn


class PRISM(nn.Module):
    def __init__(self, args,num_classes):
        super(PRISM, self).__init__()
        num_classes=args.num_items+1
        self.sub_len = 50
        emd_size=args.bert_hidden_units
        self.center=(torch.zeros(size=[num_classes,emd_size])+1e-08)
        if args.device =='cuda':
            self.center = self.center.cuda()
        self.filled_center=set()
        self.filled_center_new=set()
        self.last_target_col=None
        
    def update_center(self,inputs_row,target_row,target):
        if target is not None:
            for i in torch.unique(target):
                i=i.item()
                row_mask= (target_row==i)
                if torch.sum(row_mask) ==0:
                    continue
                self.center[i]=torch.mean(inputs_row[row_mask],dim=0)
                self.filled_center_new.add(i)
