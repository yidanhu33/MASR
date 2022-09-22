def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'
        args.use_xbm = True
        args.min_rating = 0 
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = args.batch
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.optimizer = 'Adam'
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.metric_ks = [10, 20]
        args.best_metric ='Recall@10'
        args.model_code = 'bert'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        # args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4
