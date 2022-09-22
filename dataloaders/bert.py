from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils
from collections import Counter
import random 
import numpy as np

GLOBAL_SEED = 1
from utils import fix_random_seed_as
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

class BertDataloader(AbstractDataloader):

    def __init__(self, args, dataset):
        
        super().__init__(args, dataset)
        set_seed(1)
        fix_random_seed_as(1)
        args.num_items = len(self.smap)
        args.num_users = len(self.umap)
        self.device = args.device
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder,mode = 'test')
        val_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder,mode = 'val')                                                

        self.test_negative_samples = test_negative_sampler.get_negative_samples()
        self.val_negative_samples = val_negative_sampler.get_negative_samples()
        self.train_len_sum = [len(v) for k,v in self.train.items()]

        self.train_sum = [i for k,v in  self.train.items() for i in v]
        self.val_sum = [i for k,v in  self.val.items() for i in v]
        self.test_sum = [i for k,v in  self.test.items() for i in v]
        

        self.train_sum_dic = dict(Counter(self.train_sum))
        self.total_data_sum_dic = dict(Counter(self.train_sum+ self.val_sum + self.test_sum))
        self.data_loader_num = args.data_loader_num
        args.tail_num = len(self.total_data_sum_dic) - args.head_num

        self.head_class_id = [i[0] for i in Counter(self.train_sum).most_common(args.head_num)]
        self.data_item2bilabel_dic ={}
        self.data_item2bilabel_list =[]
        for i in self.total_data_sum_dic:
            if i in self.head_class_id:
                self.data_item2bilabel_dic[i] = 1
                self.data_item2bilabel_list.append(1)
            else:
                self.data_item2bilabel_dic[i] = 0
                self.data_item2bilabel_list.append(0)

        self.data_item2bilabel_list = [-1] + self.data_item2bilabel_list

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader,val_loader_head,val_loader_tail = self._get_val_loader()
        test_loader,test_loader_head,test_loader_tail = self._get_test_loader()
        
        val_loader_list = val_loader,val_loader_head,val_loader_tail
        test_loader_list = test_loader,test_loader_head,test_loader_tail
        return train_loader, val_loader_list, test_loader_list 
    
    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng, self.data_item2bilabel_list)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')
    
    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset,dataset_head,dataset_tail = self._get_eval_dataset(mode)
        dataloader_head = data_utils.DataLoader(dataset_head, batch_size=batch_size,
                                           shuffle=False, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        dataloader_tail = data_utils.DataLoader(dataset_tail, batch_size=batch_size,
                                           shuffle=False, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True,num_workers=self.data_loader_num, worker_init_fn=worker_init_fn)
        return dataloader,dataloader_head,dataloader_tail

    def _get_eval_dataset(self, mode):
        if mode=='test':
            self.test = {k:v for k,v in self.test.items() if v[0] in self.train_sum_dic}
            answers_head = {k:v for k,v in self.test.items() if v[0] in self.head_class_id}
            answers_tail = {k:v for k,v in self.test.items() if v[0] not in self.head_class_id}
            answers = self.test
            dataset_head = BertEvalDataset(self.train, answers_head, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples,self.data_item2bilabel_list)
            dataset_tail = BertEvalDataset(self.train, answers_tail, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples,self.data_item2bilabel_list)
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples,self.data_item2bilabel_list)
        else:
         
            self.val = {k:v for k,v in self.val.items() if v[0] in self.train_sum_dic}
            answers_head = {k:v for k,v in self.val.items() if v[0] in self.head_class_id}
            answers_tail = {k:v for k,v in self.val.items() if v[0] not in self.head_class_id}
            answers = self.val

            dataset_head = BertEvalDataset(self.train, answers_head, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples,self.data_item2bilabel_list)
            dataset_tail = BertEvalDataset(self.train, answers_tail, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples,self.data_item2bilabel_list)
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.val_negative_samples,self.data_item2bilabel_list)
        return dataset,dataset_head,dataset_tail


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng, data_item2bilabel_list):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.data_item2bilabel_list = data_item2bilabel_list

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        tokens = []
        labels = []
        user_list = []
        target_bilabel=[]
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    random_i = self.rng.randint(1, self.num_items)
                    tokens.append(random_i)
                else:
                    tokens.append(s)
                labels.append(s)
                target_bilabel.append(self.data_item2bilabel_list[s]) 
            else:
                tokens.append(s)
                labels.append(0)
                target_bilabel.append(-1) #-1 is the padding for target bilabel 

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        target_bilabel = target_bilabel[-self.max_len:]
        seq= seq[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        target_bilabel = [-1] * mask_len + target_bilabel
        user_list.append(user)

        target_bilabel = torch.LongTensor(target_bilabel)
        bilabel = target_bilabel 
        if int(bilabel[bilabel>0].shape[0])==0:
            index=random.randint(0,self.__len__()-1)
            return self.__getitem__(index)
        return torch.LongTensor(tokens), torch.LongTensor(labels),bilabel
    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples,data_item2bilabel_list):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
        self.data_item2bilabel_list = data_item2bilabel_list
    def __len__(self):
        return len(self.users)
    def __getitem__(self, index):
        user = self.users[index]
        if user not in self.u2answer:
            index=random.randint(1,len(self.users)-1)
            return self.__getitem__(index)
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        bilabel = torch.LongTensor(self.data_item2bilabel_list)[answer]
        input_label = seq[:-1]+answer
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels),bilabel,torch.LongTensor(input_label)

