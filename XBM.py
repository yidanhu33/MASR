# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
class XBM:
    def __init__(self, args, memory_size):
        self.K =  memory_size
        self.feats = (torch.zeros(self.K, args.bert_hidden_units))
        self.targets = torch.zeros(self.K, dtype=torch.long)
        self.batchs = torch.zeros(self.K, dtype=torch.long)
        self.user_id = torch.zeros(self.K, dtype=torch.long)
        if args.device =='cuda':
            self.feats = self.feats.cuda()
            self.targets =  self.targets.cuda()
            self.batchs = self.batchs.cuda()
            self.user_id = self.user_id.cuda()
        self.targets[:] = -1
        self.ptr = 0
        self.neg_sampling_num=100
    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def get_pos_neg(self,target_batch):
        pos_sample = []
        pos_sample_final = []
        max_pos_len = 200
        for i in target_batch:
            i=i.item()
            row_mask= (self.targets==i)
            inputs_row_new = self.feats[row_mask]
            pos_sample.append(inputs_row_new)
        for i in  pos_sample:
            i_new = F.pad(i, pad=(0, 0, 0, max_pos_len - i.shape[0]))
            pos_sample_final.append(i_new)
        pos = torch.stack(pos_sample_final)
        neg_samples=[]
        for _ in range(self.neg_sampling_num):
            idx = np.random.choice(len(self.feats))
            while self.targets[idx] in target_batch or idx in neg_samples:
                idx = np.random.choice(len(self.feats))
            neg_samples.append(idx)
        neg=self.feats[neg_samples]
        return pos,neg
