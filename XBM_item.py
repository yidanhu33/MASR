# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import numpy as np


class XBM_Item:
    def __init__(self, args, memory_size, K):
        self.K = K
        self.item_memory_size = memory_size
        print('self.item_memory_size:',self.item_memory_size)
        self.feats = (torch.zeros(self.K, self.item_memory_size, args.bert_hidden_units))
        self.feats_emb = (torch.zeros(self.K, self.item_memory_size, args.bert_hidden_units))
        self.targets = torch.zeros(self.K,self.item_memory_size, dtype=torch.long)
        self.pointer_list = torch.zeros(args.num_items+1, dtype=torch.long)
        self.ptr = torch.zeros(self.K, dtype=torch.long)
        if args.device =='cuda':
            self.feats = self.feats.to(args.device)
            self.feats_emb = self.feats_emb.to(args.device)
            self.targets = self.targets.to(args.device)
            self.pointer_list = self.pointer_list.to(args.device)
            self.ptr = self.ptr.to(args.device)
        self.targets[:] = -1
        self.pointer_list[:]=-1
        self.ptr_class = 0
        self.neg_sampling_num=100

    def get(self):
        reshaped_targets=self.targets.view(-1)
        mask= reshaped_targets!=-1
        return self.feats.view(-1,self.feats.shape[-1])[mask], reshaped_targets[mask]#,self.feats_emb.view(-1,self.feats_emb.shape[-1])[mask]

    def enqueue_dequeue(self, feats, targets):
        if targets is not None:
            for idx,i in enumerate(targets):
                if self.pointer_list[i]==-1:
                    self.pointer_list[i]=self.ptr_class
                    self.ptr_class+=1
                pos=self.pointer_list[i]
                q_size = 1
                if self.ptr[pos] + q_size >= self.item_memory_size:
                    self.feats[pos,-q_size] = feats[idx]
                    self.targets[pos,-q_size]=i
                    self.ptr[pos] = 0
                    
                else:
                    self.feats[pos, self.ptr[pos] + q_size] = feats[idx]
                    self.targets[pos,self.ptr[pos] + q_size]=i
                    self.ptr[pos] += q_size
                    

    def get_pos_neg(self,target_batch):
        pos_idx=self.pointer_list[target_batch]
        pos=self.feats[pos_idx]
        neg_samples=[]
        for _ in range(self.neg_sampling_num):
            item = np.random.choice(len(self.feats))
            while item in target_batch or item in neg_samples:
                item = np.random.choice(len(self.feats))
            neg_samples.append(item)
        neg=self.feats[neg_samples]
        return pos,neg



        