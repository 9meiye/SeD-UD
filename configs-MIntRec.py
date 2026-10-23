from argparse import Namespace

args = Namespace(**{
    'dataset': 'MIntRec',
    'data_path': '/data/Datasets',
    'text_pretrained_model': 'bert-base-uncased',

    'text_feat_dim': 768,
    'video_feat_dim': 1024,
    'audio_feat_dim': 768,

    'text_seq_len': 30,
    'video_seq_len': 230,
    'audio_seq_len': 480,
    'max_cons_seq_length': 41,

    'num_labels': 20,
    'num_train_examples': 0,
    'train_batch_size': 4,
    'eval_batch_size': 4,
    'test_batch_size': 4,
    'num_train_epochs': 100,
    'hidden_dropout_prob': 0.3,

    'seed': 32,
    'lr': 5e-5,
    'weight_decay': 0.2,
    'pretrain_epochs': 10,
    'use_cross_attention': True,
    'fusion_heads': 8,
    'select_metric': 'accuracy',
    'idab_b2_init': -0.25,
    'idab_min_dim': 64,
    'idab_max_dim': 768,
    'classifier_hidden': 0,
    'use_pooler': True,
    'device': '',
    'save_dir': 'saved_models/MIntRec',
    'test_only': False,

    'num_workers': 8
})
