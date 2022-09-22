from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch.nn as nn
from MCL import MCL

class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.voc_len = args.num_items + 2
        self.MCL = MCL()
        self.cum_batch_num_start_M = args.cum_batch_num_start_M
        self.memory_loss_ratio_tail = args.memory_loss_ratio_tail
        self.use_memory_loss_head = args.use_memory_loss_head
        self.use_memory_loss_tail = args.use_memory_loss_tail
        self.memory_loss_ratio_head = args.memory_loss_ratio_head
    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch,cum_batch_num, xbm_mode):
        seqs, labels,bilabel = batch

        logits,batch_head_M,batch_tail_M= self.model(seqs, xbm_mode,labels,bilabel,cum_batch_num)  # B x T x V

        labels = labels.view(-1)
        bilabel= bilabel.view(-1)
        logits = logits.view(-1, logits.size(-1))
        loss = self.ce(logits, labels)
        memory_head_loss = 0
        memory_tail_loss =0
        if cum_batch_num>=self.cum_batch_num_start_M:
            if batch_tail_M is not None and self.use_memory_loss_tail:
                pos=self.model.pos_feat_membank_t
                neg=self.model.neg_feat_membank_t
                memory_tail_loss = self.MCL(batch_tail_M[0],batch_tail_M[1],pos,neg)
            if batch_head_M is not None and self.use_memory_loss_head:
                pos=self.model.pos_feat_membank_h
                neg=self.model.neg_feat_membank_h
                memory_head_loss = self.MCL(batch_head_M[0],batch_head_M[1],pos,neg)
            loss=loss+self.memory_loss_ratio_tail*memory_tail_loss+self.memory_loss_ratio_head*memory_head_loss
        return loss

    def calculate_metrics(self, batch, cum_batch_num,xbm_mode):
        seqs, candidates, labels , bilabel,input_label= batch
        scores,_,_ = self.model(seqs, xbm_mode, input_label,bilabel,cum_batch_num)  
        scores = scores[:, -1, :] 
        scores = scores.gather(1, candidates)  
        metrics_out = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics_out
    
   

    