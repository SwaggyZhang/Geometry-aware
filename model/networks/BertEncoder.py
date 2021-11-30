import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification


class BERTEncoder(nn.Module):
    def __init__(self):  # 初始化模型参数，包括tokenizer、encoder
        nn.Module.__init__(self)
        #self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, raw_token):  # 模型
        # encoded_input = self.tokenizer(raw_token,
        #                                return_tensors='pt', padding='max_length', max_length=self.max_length)
        _, sent_pooled_output = self.model(**raw_token)
        # _, label_pooled_output = self.model(**label)
        # pooled_output = alpha * sent_pooled_output + (1-alpha) * label_pooled_output
        return sent_pooled_output
