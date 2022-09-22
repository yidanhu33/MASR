from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
import os
class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader,self.val_loader_head,self.val_loader_tail = val_loader
        self.test_loader,self.test_loader_head,self.test_loader_tail = test_loader
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.pickle_name = 'ds:'+str(self.args.dataset_code)+'_lr:'+str(self.args.lr)+'_:'+str(self.args.template)
        self.writer, self.train_loggers, self.val_loggers, self.val_loggers_head, self.val_loggers_tail = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers, self.val_loggers_head, self.val_loggers_tail)
        self.log_period_as_iter = args.log_period_as_iter
        self.bert_test_template = args.template.startswith('test_bert')

        
    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        pass
    @classmethod
    @abstractmethod
    def code(cls):
        pass
    @abstractmethod
    def calculate_loss(self, batch, xbm_mode):
        pass
    @abstractmethod
    def calculate_metrics(self, batch, xbm_mode):
        pass
    def train(self):
        accum_iter = 0
        for epoch in tqdm(range(self.num_epochs)):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            # self.validate(epoch, accum_iter,code = 'head')
            # self.validate(epoch, accum_iter,code = 'tail')
            self.validate(epoch, accum_iter,code = 'overall')
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = self.train_loader
        
        for batch_idx, batch in enumerate(tqdm_dataloader):
            if batch_idx> len(tqdm_dataloader):
                break
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]
            cum_batch_num = epoch
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch,cum_batch_num, xbm_mode = 'train')
            loss.backward()
            self.optimizer.step()
   
            self.lr_scheduler.step()
            average_meter_set.update('loss', loss.item())
            accum_iter += batch_size
  
            if self._needs_to_log(accum_iter):
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)
        return accum_iter

    def validate(self, epoch, accum_iter,code):
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            if code== 'overall':
                val_dataloader = self.val_loader
                suffix = ''
            elif code== 'head':
                val_dataloader = self.val_loader_head
                suffix = '_head'
            elif code== 'tail':
                val_dataloader = self.val_loader_tail
                suffix = '_tail'
            else:
                print('error')
                exit()
 
            tqdm_dataloader = tqdm(val_dataloader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                cum_batch_num = epoch
                metrics = self.calculate_metrics(batch,cum_batch_num, xbm_mode='eval')
                for k, v in metrics.items():
                    average_meter_set.update(k+suffix, v)
                description_metrics = ['NDCG'+suffix+'@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall'+suffix+'@%d' % k for k in self.metric_ks[:3]] 
                
                description = 'Val_' +str(code)+':'+ ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG'+suffix, 'N'+suffix).replace('Recall'+suffix, 'R'+suffix)
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))

            if code== 'overall':
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_val_info(log_data)
                self.logger_service.log_val(log_data)
            elif code== 'head':
                log_data_new = {'accum_iter': accum_iter,}
                log_data_new.update(average_meter_set.averages())
                self.log_extra_val_info(log_data_new)
                self.logger_service.log_val_head(log_data_new)
            elif code== 'tail':
                log_data_new = {'accum_iter': accum_iter,}
                log_data_new.update(average_meter_set.averages())
                self.log_extra_val_info(log_data_new)
                self.logger_service.log_val_tail(log_data_new)
            else:
                print('error')
                exit()

    def test(self,code):
        print('Test best model with test set!')
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        print('self.export_root:',self.export_root)
        self.model.load_state_dict(best_model)
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            if code== 'overall':
                tqdm_dataloader = self.test_loader
            elif code== 'head':
                tqdm_dataloader = self.test_loader_head
            elif code== 'tail':
                tqdm_dataloader = self.test_loader_tail
            else:
                print('error')
                exit()
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                cum_batch_num = 1e8
                metrics = self.calculate_metrics(batch, cum_batch_num,xbm_mode='test')

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Test_' +str(code)+':'+', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))

            average_metrics = average_meter_set.averages()
            print(average_metrics)
            print('self.export_root:',self.export_root)
    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError
        
    def create_val_loggers(self,writer,model_checkpoint,code):
        if code== 'overall':
            suffix = ''
        elif code== 'head':
            suffix = '_head'
        elif code== 'tail':
            suffix = '_tail'
        else:
            print('error')
            exit()
        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k +suffix, graph_name='NDCG@%d' % k+suffix, group_name='Validation'+suffix))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k+suffix, graph_name='Recall@%d' % k+suffix, group_name='Validation'+suffix))
        if code== 'overall':
            val_loggers.append(RecentModelLogger(model_checkpoint))
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))#,Head_M = self.head_M,Tail_M = self.tail_M))
        return val_loggers

    def _create_loggers(self):
        root = Path(self.export_root)
        
        writer = SummaryWriter(root.joinpath(self.pickle_name+'logs'))
        writer.add_text('pickle_name', self.pickle_name, global_step=None, walltime=None)
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]
        val_loggers = self.create_val_loggers(writer,model_checkpoint,code = 'overall')
        val_loggers_head = self.create_val_loggers(writer,model_checkpoint,code = 'head')
        val_loggers_tail = self.create_val_loggers(writer,model_checkpoint,code = 'tail')
        return writer, train_loggers, val_loggers, val_loggers_head, val_loggers_tail

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
