from argparse import Namespace

args = Namespace(**{
    'dataset': 'MIntRec',
    'data_path': '/root/Zhang/Datasets',
    'text_pretrained_model': '/root/Zhang/Models/bert-base-uncased',

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
    'num_workers': 8
})
