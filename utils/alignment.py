import torch
from torch import nn
import torch.nn.functional as F

class CTCModule(nn.Module):
    
    def __init__(self, in_dim, out_seq_len):
        super(CTCModule, self).__init__()
        self.pred_output_position_inclu_blank = nn.LSTM(
            in_dim, out_seq_len+1, num_layers=2, batch_first=True
        )
        self.out_seq_len = out_seq_len
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        pred, _ = self.pred_output_position_inclu_blank(x)
        prob_pred = self.softmax(pred)
        
        prob_pred_output = prob_pred[:, :, 1:]
        
        prob_pred_output = prob_pred_output.transpose(1, 2)
        aligned_out = torch.bmm(prob_pred_output, x)
        
        return aligned_out


class AlignSubNet(nn.Module):
    
    def __init__(self, mode='ctc', **kwargs):
        super(AlignSubNet, self).__init__()
        self.mode = mode
        self.align_modules = nn.ModuleDict()
        
        if mode == 'ctc':
            for mod in ['text', 'video', 'audio']:
                self.align_modules[mod] = CTCModule(
                    in_dim=kwargs[f'{mod}_feat_dim'],
                    out_seq_len=kwargs['dst_len']
                )
    
    def forward(self, text_x, video_x, audio_x):
        if self.mode == 'ctc':
            text_x = self.align_modules['text'](text_x)
            video_x = self.align_modules['video'](video_x)
            audio_x = self.align_modules['audio'](audio_x)
        
        return text_x, video_x, audio_x
