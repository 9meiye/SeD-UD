import torch
import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, local_files_only=True)
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
            
        if input_ids is not None:
            input_ids = input_ids.long()
        if attention_mask is not None:
            attention_mask = attention_mask.long()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output
        
    def get_embeddings(self, input_ids):
        return self.bert.embeddings.word_embeddings(input_ids.long())
