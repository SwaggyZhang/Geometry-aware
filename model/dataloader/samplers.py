import numpy as np
import torch


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        # 无放回抽样
        self.cur_batch = 0
        self.history_cls = []

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # 返回类别 i 在label集合中的索引
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)  # 各类别样本在数据集中的索引

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # 返回一个0到len(self.m_ind)的数组，取前self.n_cls个
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class RandomSampler():

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch


# sample for each class
class ClassSampler():

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]


# for ResNet Fine-Tune, which output the same index of task examples several times
class InSetSampler():

    def __init__(self, n_batch, n_sbatch, pool):  # pool is a tensor
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch


class IncreSampler:
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        # 无放回抽样
        self.cur_batch = 0
        self.history_cls = []

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # 返回类别 i 在label集合中的索引
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)  # 各类别样本在数据集中的索引

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 返回一个0到len(self.m_ind)的数组，取前self.n_cls个或者m_ind个
            cur_classes = torch.randperm(len(self.m_ind))[:min(self.n_cls, len(self.m_ind))]
            cur_classes = torch.sort(cur_classes, descending=True)[0]
            for c in cur_classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
                del l  # 从所有类别中删除取过的
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
