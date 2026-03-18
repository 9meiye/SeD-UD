import os
import logging
import csv
import copy
import pickle
import numpy as np
import torch

from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        bm = benchmarks[args.dataset]
        max_seq_lengths, feat_dims = bm['max_seq_lengths'], bm['feat_dims']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = max_seq_lengths['text'], max_seq_lengths['video'], max_seq_lengths['audio']
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims['text'], feat_dims['video'], feat_dims['audio']
        
        self.mm_data = get_data(args)

    def save_data(self, file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            save_dict = {
                'train_data': self.mm_data['train'],
                'dev_data': self.mm_data['dev'], 
                'test_data': self.mm_data['test']
            }
            
            if file_path.endswith('.pth'):
                torch.save(save_dict, file_path)
            elif file_path.endswith('.h5'):
                import h5py
                with h5py.File(file_path, 'w') as f:
                    for key, value in save_dict.items():
                        if isinstance(value, (dict, list, str, int, float)):
                            f.attrs[key] = str(value)
                        else:
                            pass
            else:
                raise ValueError("Only .pth and .h5 formats are supported")
                
            self.logger.info(f"Data saved to {file_path}")
        
def get_data(args):
    
    if args.dataset in ['mosi', 'mosei']:
        return get_emt_dlfr_data(args)
    else:
        return get_original_data(args)

def get_emt_dlfr_data(args):
    data_path = os.path.join(args.data_path, 'cmu-mosi' if args.dataset == 'mosi' else args.dataset)
    bm = benchmarks[args.dataset]
    
    data_files = [
        os.path.join(data_path, f'{args.dataset}_data.pkl'),
        os.path.join(data_path, 'data.pkl'),
        os.path.join(data_path, 'mosi_data.pkl'),
    ]
    
    data_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_file = file_path
            break
    
    if data_file is None:
        train_file = os.path.join(data_path, 'train.pkl')
        dev_file = os.path.join(data_path, 'dev.pkl')
        test_file = os.path.join(data_path, 'test.pkl')
        
        if all(os.path.exists(f) for f in [train_file, dev_file, test_file]):
            return load_separate_files(args, train_file, dev_file, test_file)
        else:
            raise FileNotFoundError(f"Dataset file not found for {args.dataset}, tried: {data_files}")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    train_text = data['train']['text'].astype(np.float32)
    train_vision = data['train']['vision'].astype(np.float32)
    train_audio = data['train']['audio'].astype(np.float32)
    train_labels = data['train']['labels'].astype(np.float32)
    
    dev_text = data['valid']['text'].astype(np.float32)
    dev_vision = data['valid']['vision'].astype(np.float32)
    dev_audio = data['valid']['audio'].astype(np.float32)
    dev_labels = data['valid']['labels'].astype(np.float32)
    
    test_text = data['test']['text'].astype(np.float32)
    test_vision = data['test']['vision'].astype(np.float32)
    test_audio = data['test']['audio'].astype(np.float32)
    test_labels = data['test']['labels'].astype(np.float32)
    
    train_label_ids = (train_labels > 0).astype(int)
    dev_label_ids = (dev_labels > 0).astype(int)
    test_label_ids = (test_labels > 0).astype(int)
    
    train_video_data = {
        'feats': train_vision,
        'lengths': [train_vision.shape[1]] * len(train_vision)
    }
    train_audio_data = {
        'feats': train_audio,
        'lengths': [train_audio.shape[1]] * len(train_audio)
    }
    
    dev_video_data = {
        'feats': dev_vision,
        'lengths': [dev_vision.shape[1]] * len(dev_vision)
    }
    dev_audio_data = {
        'feats': dev_audio,
        'lengths': [dev_audio.shape[1]] * len(dev_audio)
    }
    
    test_video_data = {
        'feats': test_vision,
        'lengths': [test_vision.shape[1]] * len(test_vision)
    }
    test_audio_data = {
        'feats': test_audio,
        'lengths': [test_audio.shape[1]] * len(test_audio)
    }
    
    train_data = MMDataset(train_label_ids, train_text, train_video_data, train_audio_data)
    dev_data = MMDataset(dev_label_ids, dev_text, dev_video_data, dev_audio_data)
    test_data = MMDataset(test_label_ids, test_text, test_video_data, test_audio_data)

    data = {'train': train_data, 'dev': dev_data, 'test': test_data}
    return data

def load_separate_files(args, train_file, dev_file, test_file):
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(dev_file, 'rb') as f:
        dev_data = pickle.load(f)
    
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    train_text = train_data['text_bert'].astype(np.float32) if 'text_bert' in train_data else train_data['text'].astype(np.float32)
    train_vision = train_data['vision'].astype(np.float32)
    train_audio = train_data['audio'].astype(np.float32)
    train_labels = train_data['regression_labels'].astype(np.float32) if 'regression_labels' in train_data else train_data['classification_labels'].astype(np.float32)
    
    dev_text = dev_data['text_bert'].astype(np.float32) if 'text_bert' in dev_data else dev_data['text'].astype(np.float32)
    dev_vision = dev_data['vision'].astype(np.float32)
    dev_audio = dev_data['audio'].astype(np.float32)
    dev_labels = dev_data['regression_labels'].astype(np.float32) if 'regression_labels' in dev_data else dev_data['classification_labels'].astype(np.float32)
    
    test_text = test_data['text_bert'].astype(np.float32) if 'text_bert' in test_data else test_data['text'].astype(np.float32)
    test_vision = test_data['vision'].astype(np.float32)
    test_audio = test_data['audio'].astype(np.float32)
    test_labels = test_data['regression_labels'].astype(np.float32) if 'regression_labels' in test_data else test_data['classification_labels'].astype(np.float32)
    
    if len(train_labels.shape) == 1:
        train_label_ids = (train_labels > 0).astype(int)
        dev_label_ids = (dev_labels > 0).astype(int)
        test_label_ids = (test_labels > 0).astype(int)
    else:
        train_label_ids = train_labels.argmax(axis=1) if len(train_labels.shape) > 1 else train_labels
        dev_label_ids = dev_labels.argmax(axis=1) if len(dev_labels.shape) > 1 else dev_labels
        test_label_ids = test_labels.argmax(axis=1) if len(test_labels.shape) > 1 else test_labels
    
    train_data = MMDataset(train_label_ids, train_text, train_vision, train_audio)
    dev_data = MMDataset(dev_label_ids, dev_text, dev_vision, dev_audio)
    test_data = MMDataset(test_label_ids, test_text, test_vision, test_audio)

    data = {'train': train_data, 'dev': dev_data, 'test': test_data}
    return data

def get_original_data(args):
    data_path = os.path.join(args.data_path, args.dataset)
    bm = benchmarks[args.dataset]
    
    label_list = copy.deepcopy(bm["labels"])
      
    args.num_labels = len(label_list)  
    
    
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'train.tsv'))
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'dev.tsv'))
    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'test.tsv'))
    args.num_train_examples = len(train_data_index)

    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
        'bm': bm,
    }

    data_args['max_seq_len'] = args.text_seq_len = bm['max_seq_lengths']['text']    
    text_data = get_t_data(args, data_args)
        
    video_feats_path = os.path.join(data_path, 'video_data/video_feats.pkl')
    video_feats_data_args = {
        'data_path': video_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    video_feats_data_args['max_seq_len'] = args.video_seq_len = bm['max_seq_lengths']['video']
    video_feats_data = get_v_a_data(video_feats_data_args, video_feats_path)

    audio_feats_path = os.path.join(data_path, 'audio_data/audio_feats.pkl')
    audio_feats_data_args = {
        'data_path': audio_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    audio_feats_data_args['max_seq_len'] = args.audio_seq_len = bm['max_seq_lengths']['audio']
    audio_feats_data = get_v_a_data(audio_feats_data_args, audio_feats_path)
    

    train_data = MMDataset(train_label_ids, text_data['train'], video_feats_data['train'], audio_feats_data['train'])
    dev_data = MMDataset(dev_label_ids, text_data['dev'], video_feats_data['dev'], audio_feats_data['dev'])
    test_data = MMDataset(test_label_ids, text_data['test'], video_feats_data['test'], audio_feats_data['test'])

    data = {'train': train_data, 'dev': dev_data, 'test': test_data}

    return data
                 
def get_indexes_annotations(args, bm, label_list, read_file_path):

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    with open(read_file_path, 'r', encoding='utf-8') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if args.dataset in ['MIntRec']:
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                
                label_id = label_map[line[4]]
            
            elif args.dataset in ['MELD-DA']:
                label_id = label_map[line[3]]
                
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
            
            elif args.dataset in ['IEMOCAP-DA']:
                label_id = label_map[line[2]]
                index = line[0]
                indexes.append(index)
            
            label_ids.append(label_id)
    
    return indexes, label_ids
