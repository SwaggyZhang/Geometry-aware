import torch
import os.path as osp

from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import BertTokenizer
import numpy as np
# TODO: 修改数据集为OOS
THIS_PATH = osp.dirname(__file__)  # 返回当前路径
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
#IMAGE_PATH1 = osp.join('./', 'data/miniimagenet/images')
SPLIT_PATH = osp.join('./', 'data/OOS')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')


def identity(x):
    return x


class OOS(Dataset):
    """ Usage:
    """

    def __init__(self, setname, args, max_length=40, augment=False,):
        csv_path = osp.join(SPLIT_PATH, setname + '.txt')

        # TODO: 加入encoder，对数据编码，便于后续放入cuda中
        self.data, self.label_ind, self.label = self.parse_csv(csv_path, setname)  # 读数据名、标签名
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_class = len(set(self.label))  # 统计类别数目
        self.max_length = max_length
# 读CSV文件

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r', encoding='utf-8').readlines()]

        data = []
        label_ind = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            # wnid为类别名，sentence为句子
            wnid, sentence = l.split('\t')
            #sentence = sentence.split(' ')

            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            # 句子
            data.append(sentence)
            label.append(wnid)
            # 类别索引
            label_ind.append(lb)
        # 返回句子列表、标签集合
        return data, label_ind, label

    def __len__(self):  # 获取样本数量
        return len(self.data)

    def __getitem__(self, i):  # 获取样本、标签
        input_data, label = self.data[i], self.label[i]

        return input_data, label
