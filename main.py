
from options import args
from models.__init__ import model_factory
from dataloaders.__init__ import dataloader_factory
from trainers.__init__ import trainer_factory
from utils import *
from XBM import XBM
from XBM_item import XBM_Item
from centroids_set import PRISM
def train():
    
    print(args.device)
    export_root = setup_train(args)
    
    train_loader, val_loader, test_loader= dataloader_factory(args)
    print('args.num_users+1:',args.num_users+1)
    print('args.num_items+1:',args.num_items+1)
    xbm_f = XBM(args,args.XBM_SIZE_HEAD)
    centroids_f = PRISM(args,args.num_items+1)  
    sub_memory_size = args.item_memory_size_change
    xbm_t = XBM_Item(args,sub_memory_size,args.tail_num)
    model = model_factory(args,xbm_f,centroids_f,xbm_t) 

    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()


    model.iteration=0 
    trainer.test(code='overall')
    model.iteration=0 
    trainer.test(code='head')
    model.iteration=0 
    trainer.test(code='tail')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
