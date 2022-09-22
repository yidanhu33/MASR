from .base import BaseModel
from .bert_modules.bert import BERT
import torch.nn as nn
import torch.nn.functional as F
import torch 
class BERTModel(BaseModel):
    def __init__(self,args,xbm_f,centroids_f,xbm_t): 
        super().__init__(args)
        self.xbm_f = xbm_f
        self.xbm_t = xbm_t
        self.centroids_f = centroids_f
        self.copy_head_label = args.copy_head_label
        self.copy_tail_label = args.copy_tail_label
        self.top_k_item_head = args.top_k_item_head
        self.top_k_item_tail = args.top_k_item_tail
        self.use_memory_loss_head = args.use_memory_loss_head
        self.use_memory_loss_tail = args.use_memory_loss_tail
        self.cum_batch_num_start_M = args.cum_batch_num_start_M
        
        self.sigmoid = nn.Sigmoid()
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 2)
        self.out_head = nn.Linear(self.bert.hidden, args.num_items + 2)
        self.out_tail = nn.Linear(self.bert.hidden, args.num_items + 2)
        self.bi_out_linear = nn.Linear((args.num_items + 2)*2, 1)
        self.bi_out_linear_new = nn.Linear(self.bert.hidden, 1)
        self.head_copy_layer = nn.Linear(self.bert.hidden*2, 1)
    
        if self.copy_head_label or self.copy_tail_label:
            self.bilabel_embeddings =  self.bert.embedding.token

        
    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x_seqs, xbm_mode, labels_ori,bilabel,cum_batch_num):
        x = self.bert(x_seqs) #[B,S,W]
        batch_size = x.shape[0]
        hidden_state = x.shape[-1]
        feas = x.reshape(-1,x.shape[-1])#[B*S,W]
        if xbm_mode=='train':
            labels = labels_ori.view(-1) #[B*S]
            bilabel = bilabel.view(-1) #[B*S]
            tail_bilabel_mask = bilabel==0
            tail_bilabel_mask = tail_bilabel_mask.view(-1) 
            tail_fea = feas[tail_bilabel_mask]
            tail_fea = F.normalize(tail_fea,p=2,dim=-1)
            tail_fea_labels = labels[tail_bilabel_mask]
            head_bilabel_mask = bilabel==1
            head_bilabel_mask = head_bilabel_mask.view(-1) 
            head_fea = feas[head_bilabel_mask]
            head_fea_labels = labels[head_bilabel_mask]
            head_fea = F.normalize(head_fea,p=2,dim=-1)
            if self.use_memory_loss_tail:
                self.pos_feat_membank_t,self.neg_feat_membank_t = self.xbm_t.get_pos_neg(tail_fea_labels)
                self.pos_feat_membank_t=self.pos_feat_membank_t.clone() 
            if self.use_memory_loss_head:
                self.pos_feat_membank_h,self.neg_feat_membank_h = self.xbm_f.get_pos_neg(head_fea_labels)
                self.pos_feat_membank_h=self.pos_feat_membank_h.clone()  
        batch_head_M, batch_tail_M = None, None
        if  cum_batch_num<self.cum_batch_num_start_M:
            if xbm_mode=='train':
                self.xbm_f.enqueue_dequeue(head_fea.detach(),head_fea_labels.detach())
                xbm_feats, xbm_targets = self.xbm_f.get()
                self.centroids_f.update_center(xbm_feats.detach(),xbm_targets.detach(),head_fea_labels.detach())
                self.xbm_t.enqueue_dequeue(tail_fea.detach(),tail_fea_labels.detach())
                xbm_t_tail_feats,xbm_t_tail_targets = self.xbm_t.get()
            return self.out(x),batch_head_M,batch_tail_M
        else:

            with torch.no_grad(): 
                xbm_t_tail_feats,xbm_t_tail_targets = self.xbm_t.get()
                norm_x_new = torch.norm(feas.clone(), 2, -1, keepdim=True)
                cur_x_new =  feas / (norm_x_new+1e-08) #[B*S,W]
                top_k_tail_feas = torch.mm(cur_x_new, xbm_t_tail_feats.t()) #[B*S,500]
                top_k_tail_feas_v_o, top_k_tail_feas_idx= torch.topk(top_k_tail_feas, self.top_k_item_tail) #[B*S,10]
                top_k_tail_feas_v = F.softmax(top_k_tail_feas_v_o,dim = -1)
                top_k_tail_feas_v = top_k_tail_feas_v.unsqueeze(-1)  #[B*S,10,1]
                x_tail_label = xbm_t_tail_targets[top_k_tail_feas_idx]
                x_tail_label = x_tail_label.reshape(feas.shape[0],-1)#[B*S,10]
               
                if self.copy_tail_label:
                    x_tail_label = self.bilabel_embeddings(x_tail_label)
                    x_tail_label = x_tail_label*top_k_tail_feas_v
                    x_tail_label = torch.sum(x_tail_label,dim = -2)
                else:
                    x_tail_feas = xbm_t_tail_feats[top_k_tail_feas_idx] #[B*S*10,W]
                    x_tail_feas = x_tail_feas.reshape(feas.shape[0],-1,x_tail_feas.shape[-1])
                    x_tail_feas = x_tail_feas*top_k_tail_feas_v
                    x_tail_feas = torch.mean(x_tail_feas,dim = -2)
            if self.copy_tail_label:
                top_k_tail_feas_v_out = self.out_tail(x_tail_label)#[B*S,V]
            else:
                top_k_tail_feas_v_out = self.out_tail(x_tail_feas)#[B*S,V]
            top_k_tail_feas_v_out = top_k_tail_feas_v_out.reshape(batch_size,-1,top_k_tail_feas_v_out.shape[-1])
            x_out = self.out(x)
  
            with torch.no_grad(): 
                xbm_feats, xbm_targets = self.xbm_f.get()
                batch_size = x.shape[0]
                centroids_f_center_mean = self.centroids_f.center #[6041,W]
                cos_head_feas = torch.mm(cur_x_new, centroids_f_center_mean.t())
                top_k_head_feas_v, top_k_head_feas_idx= torch.topk(cos_head_feas, self.top_k_item_head) #[B*S,10]
                top_k_head_feas_v = top_k_head_feas_v.unsqueeze(-1)#.unsqueeze(-1)   #[B*S,10,1]
                head_feas = centroids_f_center_mean[top_k_head_feas_idx] #[B*S,W]
                head_feas = head_feas.reshape(feas.shape[0],-1,hidden_state)
                head_feas = head_feas*top_k_head_feas_v
                head_feas = torch.mean(head_feas,dim = -2)

            head_feas = head_feas.reshape(batch_size,-1,hidden_state)
            head_copy = torch.cat((x,head_feas),dim = -1)
            head_copy = self.head_copy_layer(head_copy)
            head_copy = self.sigmoid(head_copy)
            
            if self.copy_head_label:
                with torch.no_grad(): 
                    x_head_label = top_k_head_feas_idx.reshape(feas.shape[0],-1)#[B*S,10]
                    x_head_label = self.bilabel_embeddings(x_head_label)
                    x_head_label = x_head_label*top_k_head_feas_v
                    x_head_label = torch.sum(x_head_label,dim = -2)
                x_head_label = x_head_label.reshape(batch_size,-1,hidden_state)
                head_out = self.out_head(x_head_label)#[B*S,V]
            else:
                head_out = self.out_head(head_feas)##[B,3*50,V]
            head_out_final = (1-head_copy)*head_out + head_copy*x_out

            bi_out = self.bi_out_linear(torch.cat([head_out_final,top_k_tail_feas_v_out],dim=-1))
            bi_out_new = self.sigmoid(bi_out)

            out= bi_out_new*head_out_final +(1-bi_out_new)*top_k_tail_feas_v_out
            if xbm_mode=='train':
                self.xbm_t.enqueue_dequeue(tail_fea.detach(),tail_fea_labels.detach())
                self.xbm_f.enqueue_dequeue(head_fea.detach(), head_fea_labels.detach())
                self.centroids_f.update_center(xbm_feats.detach(),xbm_targets.detach(),head_fea_labels.detach())
                batch_head_M = head_fea,head_fea_labels
                batch_tail_M = tail_fea,tail_fea_labels
            return  out,batch_head_M,batch_tail_M
        


