# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    PrototypicalBatchSampler：在每次迭代时生成一批索引。
    指数的计算方法是将“classes_per_it”和“num_samples”记入账户，
    事实上，在每次迭代中，批处理索引都会引用“num_support”+“num_query”样本
    对于“classes_per_it”随机类。

    __len__ 返回每个epoch的集数（与“self.iterations”相同）。
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        labels: 数据集中所有样本的标签,用于根据类别采样数据
        classes_per_it: 每个批次中包含的不同类别数
        num_samples: 每个类别的样本数量,包括support和query样本
        iterations: 每个epoch的迭代次数或批次数

        labels: 这是一个列表或数组，包含当前数据集中所有数据点的标签。标签用于按类别对数据点进行分组和采样。
        classes_per_it: 这是一个整数，指定每次迭代采样的不同类别的数量。例如，如果classes_per_it是5，那么每次迭代将包含从数据集中随机选择的5个类别。
        num_samples: 这是一个整数，指定每个类别在每次迭代中的样本数量。样本分为支持集和查询集，其中支持集用于计算每个类别的原型表示，查询集用于评估分类性能。
        例如，如果num_samples是10，那么每个类别将有10个样本，可以分为5个支持和5个查询样本。
        iterations: 这是一个整数，指定每个epoch生成的迭代（或episode）的数量。一个epoch是对数据集的完整遍历，一个迭代是一个数据批次，
        包含classes_per_it个类别和每个类别num_samples个样本。例如，如果iterations是100，那么每个epoch将有100个迭代或数据批次。
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        if self.labels.dtype.type is np.str_:
            self.labels = self.labels.astype(np.unicode_)
            self.labels = self.labels.astype(np.int64)
        "调用父类构造函数初始化,并处理标签数据的类型,确保其为整数类型。"
        # print(labels,len(labels))
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        "将传入的参数保存为类的属性,方便后续使用。"
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        if self.classes.dtype.type is np.str_:
            self.classes = self.classes.astype(np.unicode_)
            self.classes = self.classes.astype(np.int64)
        self.classes = torch.LongTensor(self.classes)
        "获取数据集中所有不同的类别self.classes以及每个类别的样本数量self.counts,同样确保类别标签为整数类型,并转换为PyTorch的LongTensor。"

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        '''初始化一些变量:
        self.idxs是所有样本的索引
        self.indexes是一个矩阵,行数为类别数,列数为最大的类别样本数,初始化为nan
        self.numel_per_class记录每个类别的实际样本数'''
        for idx, label in enumerate(self.labels):
            # print((self.classes == label).numpy().astype(int))
            label_idx = np.argwhere((self.classes == label).numpy().astype(int)).item()
            # print(label_idx)
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1
        "遍历所有标签,填充self.indexes矩阵和self.numel_per_class。对于每个样本,"
        "找到它所属的类别索引label_idx,然后在self.indexes的对应行中找一个空位置填入该样本的索引,"
        "并将该类别的样本数加1。"

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it
        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            "每个批次的大小为batch_size,等于每个类别的样本数spc乘以类别数cpi。先随机选出cpi个类别索引。"
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
                "对于选出的每个类别,从self.indexes中选出spc个样本的索引,放入batch对应的位置。"
            batch = batch[torch.randperm(len(batch))]
            yield batch
            "最后打乱batch中样本的顺序,将其作为本次迭代的结果yield出去。"

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        返回每个epoch包含的迭代次数或批次数。
        '''
        return self.iterations
