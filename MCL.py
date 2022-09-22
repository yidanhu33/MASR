from torch import nn
import torch
class MCL(nn.Module):
    def __init__(self):
        super(MCL, self).__init__()
        self.margin = 0.5
    def forward(self, feats_batch, targets_batch, pos_memoryBank, neg_memoryBank):
        n = feats_batch.size(0)
        d=feats_batch.size(-1)
        neg_memoryBank=neg_memoryBank.view(-1,d)
        pos_sim_mat = torch.bmm(feats_batch.view(n,1,d), pos_memoryBank.permute([0,2,1]))
        neg_sim_mat = torch.matmul(feats_batch, neg_memoryBank.t()).view(-1)
        loss = list()
        neg_count = list()
        all_pos_loss=[]
        all_neg_loss=[]
        neg_pair = torch.masked_select(neg_sim_mat, neg_sim_mat > self.margin)
        pos_loss = torch.sum(-pos_sim_mat + 1)
        if len(neg_pair) > 0:
            neg_loss = torch.sum(neg_pair)
            neg_count.append(len(neg_pair))
        else:
            neg_loss = torch.tensor(0)
        loss.append(pos_loss + neg_loss)
        all_pos_loss.append(pos_loss.item())
        all_neg_loss.append(neg_loss.item())
        prefix = "memory_"
        log_info={f'{prefix}_pos_loss':sum(all_pos_loss)/n,f'{prefix}_neg_loss':sum(all_neg_loss)/n}
        if len(neg_count) != 0:
            log_info[f"{prefix}average_neg"] = sum(neg_count) / len(neg_count)
        else:
            log_info[f"{prefix}average_neg"] = 0
        loss = sum(loss) / n  
        return loss