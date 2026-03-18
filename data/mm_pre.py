from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']


    
class MMDataset(Dataset):
    def __init__(self, labels, text_data, video_data, audio_data):
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.label_ids = labels
        self.size = len(self.text_data)

    def __len__(self):
        return self.size

    def __getitem__(self, index):    
        sample = {
            'text_feats': torch.tensor(self.text_data[index], dtype=torch.long),
            'video_feats': torch.tensor(self.video_data['feats'][index], dtype=torch.float).squeeze(),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(self.audio_data['feats'][index], dtype=torch.float),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index])),
        }
        if self.label_ids is not None:
            sample['label_ids'] = torch.tensor(self.label_ids[index], dtype=torch.long)
            sample['label_ids'] = sample['label_ids'].clone().detach()
        return sample
