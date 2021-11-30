import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
maxlength = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_length = 5

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        elif args.backbone_class == 'BERT' or 'BERTLabel':
            hdim = 768
            from model.networks.BertEncoder import BERTEncoder
            self.encoder = BERTEncoder()  # set the max_length 30
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, label=None, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction

            x = tokenizer(x, return_tensors='pt', padding='max_length', truncation=True, max_length=maxlength)
            x = x.to(device)
            if label is not None:
                label = tokenizer(label, return_tensors='pt', padding='max_length', truncation=True,
                              max_length=label_length)
                label = label.to(device)
                label_embs = self.encoder(label)
            instance_embs = self.encoder(x)  # vector from bert
            # num_inst = instance_embs.shape[0]
            # inst_size = instance_embs.shape  # shape of tensor (batch_size, max_length, word_dim)
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)

            if self.training:
                # TODO: feat_cocat.py _foward()加入label_emb
                if label:
                    logits, logits_reg, proto_reg = self._forward(instance_embs, support_idx, query_idx, label_embs)  # input the data in feat
                    return logits, logits_reg, proto_reg
                else:
                    logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, label, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')