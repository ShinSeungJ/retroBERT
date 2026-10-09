"""retroBERT: BERT encoder over pose-keypoint sequences."""

import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

class retroBERT(nn.Module):

    def __init__(self, input_dim, max_seq_length, bert_dim=768):
        super(retroBERT, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, bert_dim))
        self.embedding = self.BERTembedding(input_dim, bert_dim)

        bare_config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=2,
        )

        self.bert = BertForSequenceClassification(bare_config)

    def PositionalEncoding(self, max_seq_length, bert_dim=768, device=None):
        # Built on the input's device so the model follows .to(), rather than being
        # pinned to whatever was available when it was constructed.
        if device is None:
            device = self.cls_token.device
        pe = torch.zeros(max_seq_length, bert_dim, device=device)
        position = torch.arange(0, max_seq_length, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, bert_dim, 2, device=device).float() * -(np.log(10000.0) / bert_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def BERTembedding(self, input_dim, bert_dim=768):
        embedding = nn.Linear(input_dim, bert_dim)
        return embedding

    def forward(self, x, attention_mask=None, labels=None):
        x = self.embedding(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        positional_encodings = self.PositionalEncoding(x.size(1), x.size(2), x.device)
        embedded_x = x + positional_encodings
        outputs = self.bert(inputs_embeds=embedded_x, attention_mask=attention_mask, labels=labels)
        return outputs
