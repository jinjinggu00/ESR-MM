import csv
import fcntl
import os
import pickle
import random

import numpy as np
import pynvml
import time
import torch


def getAvaliableDevice(gpu=[0], min_mem=24000, left=False):
    # def getAvaliableDevice(gpu=[6],min_mem=10000,left=False):
    """
    :param gpu:
    :param min_mem:
    :param left:
    :return:
    """

    pynvml.nvmlInit()
    t = int(time.strftime("%H", time.localtime()))

    if t >= 23 or t < 8:
        left = False  # do not leave any GPUs
    # else:
    # left=True

    min_num = 3
    dic = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, -1: -1}  # just for 207 server
    ava_gpu = -1

    while ava_gpu == -1:
        avaliable = []
        for i in gpu:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            if (meminfo.free / 1024 ** 2) > min_mem and utilization.gpu < 10:
                avaliable.append(dic[i])

        if len(avaliable) == 0 or (left and len(avaliable) <= 1):
            # if len(avaliable)==1:
            #     if avaliable[0] not in [4,5,6]:
            #         ava_gpu= -1
            #         time.sleep(5)
            #         continue
            # else :
            ava_gpu = -1
            time.sleep(20)
            continue
        ava_gpu = avaliable[0]
    return ava_gpu


def write_shared_file(file_name, content):
    nowtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    content[0] = nowtime + " " + content[0]
    with open(file_name, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.writelines(content)
        fcntl.flock(f, fcntl.LOCK_UN)


def write_csv_file(file_name, content):
    nowtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    content["time"] = nowtime
    to_write_head = False
    if not os.path.exists(file_name):
        to_write_head = True
    with open(file_name, 'a+') as f:
        writer = csv.DictWriter(f, content.keys())
        fcntl.flock(f, fcntl.LOCK_EX)
        if to_write_head:
            writer.writeheader()
        writer.writerow(content)
        # for key, value in content.items:
        #     writer.writerow([key, value])
        fcntl.flock(f, fcntl.LOCK_UN)


def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # np.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def manhattan_dist(x, y):
    """
    Calculates the Manhattan distance between two tensors.

    Args:
        x (torch.Tensor): First tensor of shape (n, d).
        y (torch.Tensor): Second tensor of shape (m, d).

    Returns:
        torch.Tensor: Manhattan distance matrix of shape (n, m).
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.abs(x - y).sum(dim=2)

    return dist


def cosine_dist(x1, x2):
    """
    Compute cosine distance between two tensors.

    Args:
        x1 (torch.Tensor): First tensor of shape (n1, d).
        x2 (torch.Tensor): Second tensor of shape (n2, d).

    Returns:
        dist (torch.Tensor): Cosine distance tensor of shape (n1, n2).
    """
    x1_norm = x1 / x1.norm(dim=1, keepdim=True)
    x2_norm = x2 / x2.norm(dim=1, keepdim=True)
    dist = 1 - torch.mm(x1_norm, x2_norm.t())
    return dist


def euclidean_dist(x, y):
    '''

        欧氏_dist函数以两个张量作为输入：x和y。x张量的形状为（N，D），其中N是样本数，D是特征数。
        y张量的形状为（M，D），其中M是样本数，D是特征数。该函数假设两个张量具有相同数量的特征。

        该函数首先获取两个张量的大小，并将其分配给变量n、m和d。然后，通过使用assert语句检查这两个张量是否具有相同数量的特征。

        然后，函数通过在索引1处添加新的维度并沿着该维度重复张量M次，将x张量扩展为具有（N，M，D）的形状。
        它还通过在索引0处添加新的维度并沿该维度重复张量N次，将y张量扩展为具有（N，M，D）的形状。

        然后，该函数计算两个张量之间的元素差，并将其平方。然后，它将最后一个维度（D）上的平方差相加，并返回结果张量。
        这个张量的形状为（N，M），表示从x到y的每对样本之间的欧几里得距离。
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def chebyshev_dist(x, y):
    """
    Calculates the Chebyshev distance between two tensors.
    切比雪夫距离
    Args:
        x (torch.Tensor): First tensor of shape (n, d).
        y (torch.Tensor): Second tensor of shape (m, d).

    Returns:
        torch.Tensor: Chebyshev distance matrix of shape (n, m).
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.max(torch.abs(x - y), dim=2)[0]

    return dist


def jaccard_dist(x, y):
    """
    Calculates the Jaccard distance between two tensors.
    杰卡德距离
    Args:
        x (torch.Tensor): First tensor of shape (n, d).
        y (torch.Tensor): Second tensor of shape (m, d).

    Returns:
        torch.Tensor: Jaccard distance matrix of shape (n, m).
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    intersection = torch.min(x, y).sum(dim=2)
    union = torch.max(x, y).sum(dim=2)

    dist = 1 - (intersection / union)

    return dist


def hyperbolic_distance(x, y):
    """
    Calculates the hyperbolic distance between two sets of points in the Poincaré ball model.

    Args:
        x (torch.Tensor): First set of points with shape (batch_size, dim).
        y (torch.Tensor): Second set of points with shape (num_classes, dim).

    Returns:
        torch.Tensor: Hyperbolic distance matrix with shape (batch_size, num_classes).
    """
    # 确保 x 和 y 的范数小于 1
    x = x / (1 + 1e-5)
    y = y / (1 + 1e-5)
    # Expand y to match the batch size of x
    y = y.unsqueeze(0).expand(x.size(0), -1, -1)

    # Calculate the Euclidean norm of the points
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=2, keepdim=True)


    # Ensure norms are broadcastable, focusing on the correct dimensions
    x_norm_sq = x_norm.pow(2)
    y_norm_sq = y_norm.pow(2).transpose(1, 2)

    # Calculate the squared Euclidean distance in a broadcast-compatible way
    numerator = 2 * torch.sum((x.unsqueeze(1) - y) ** 2, dim=2)

    denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
    # 为了数值稳定性，确保分母不为0，并且 acosh 的参数不小于 1
    denominator = torch.clamp(denominator, min=1e-5)
    acosh_param = 1 + numerator / denominator.squeeze()
    acosh_param = torch.clamp(acosh_param, min=1.0)
    # Compute the Poincaré distance ensuring the shapes align
    poincare_dist = torch.acosh(acosh_param)

    return poincare_dist


def get_support_query_data(support, query, device):
    '''
    :param support:[n_class, c, v]
    :param query: [n_class * n_query, c, v]
    :return: sq: [n_class * (n_class * n_query) * 2, c, v]
    '''
    n_class, c, v = support.size()
    all_query = query.size(0)
    sum_matching_graph = n_class * all_query * 2

    node_features = torch.zeros(sum_matching_graph, c, v).to(device)

    idx, idx2 = torch.arange(0, sum_matching_graph, 2).to(device), torch.arange(1, sum_matching_graph, 2).to(device)
    node_features[idx] = query.unsqueeze(1).repeat(1, n_class, 1, 1).reshape(-1, c, v)
    node_features[idx2] = support.unsqueeze(0).repeat(all_query, 1, 1, 1).reshape(-1, c, v)

    node_features = node_features.permute(0, 2, 1).reshape(sum_matching_graph * v, c)

    return node_features


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)


def compute_similarity(x, y):
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """

    return -euclidean_distance(x, y)


def check_nan(tensor):
    '''
    Check if there is any NaN in the tensor
    if check_nan(dist):
        print("The tensor contains nan values.")

    '''
    return torch.isnan(tensor).any().item()


def extract_k_segement(x, num_frame, segement):
    n, c, t, v = x.size()

    assert n == len(num_frame)
    step = num_frame // segement

    new_x = []
    for i in range(n):
        idx = [random.randint(j * step[i], (j + 1) * step[i] - 1) for j in range(segement)]
        new_x.append(x[i, :, idx, :].unsqueeze(0))

    new_x = torch.cat(new_x, dim=0)

    return new_x


def load_data(path, train_class_name, val_class_name, test_class_name):
    data_path = os.path.join(path, 'train_data.npy')
    label_path = os.path.join(path, 'train_label.pkl')
    # num_frame_path = os.path.join(path, 'train_num_frame.npy')
    num_class = np.zeros(125)

    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')

    # load data
    data = np.load(data_path)
    # num_frame = np.load(num_frame_path)
    num_frame = np.ones(len(label)) * 300

    train_data, val_data, test_data = [], [], []
    train_label, val_label, test_label = [], [], []
    train_num_frame, val_num_frame, test_num_frame = [], [], []
    for i in range(len(label)):

        if label[i] > 120:
            continue

        num_class[label[i]] += 1

        if label[i] in train_class_name:
            if num_class[label[i]] >= 500:
                continue
            train_data.append(np.expand_dims(data[i], axis=0))
            train_label.append(label[i])
            train_num_frame.append(num_frame[i])
        elif label[i] in val_class_name:
            if num_class[label[i]] >= 100:
                continue
            val_data.append(np.expand_dims(data[i], axis=0))
            val_label.append(label[i])
            val_num_frame.append(num_frame[i])
        elif label[i] in test_class_name:
            if num_class[label[i]] >= 100:
                continue
            test_data.append(np.expand_dims(data[i], axis=0))
            test_label.append(label[i])
            test_num_frame.append(num_frame[i])
    train_data, val_data, test_data = np.concatenate(train_data, 0), np.concatenate(val_data, 0), np.concatenate(
        test_data, 0)

    save_path = '/mnt/data1/kinetics-skeleton/train_500_val_100'

    np.save(os.path.join(save_path, 'train_data.npy'), train_data)
    np.save(os.path.join(save_path, 'train_label.npy'), train_label)
    np.save(os.path.join(save_path, 'train_frame.npy'), train_num_frame)
    np.save(os.path.join(save_path, 'val_data.npy'), val_data)
    np.save(os.path.join(save_path, 'val_label.npy'), val_label)
    np.save(os.path.join(save_path, 'val_frame.npy'), val_num_frame)
    np.save(os.path.join(save_path, 'test_data.npy'), test_data)
    np.save(os.path.join(save_path, 'test_label.npy'), test_label)
    np.save(os.path.join(save_path, 'test_frame.npy'), test_num_frame)

    data_list = [train_data, train_label, np.array(train_num_frame), val_data, val_label, np.array(val_num_frame),
                 test_data, test_label, np.array(test_num_frame)]

    return data_list


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 距离矩阵归一化函数
def dist_normalization(dist):
    dist_1_min, _ = torch.min(dist, dim=1, keepdim=True)  # 计算每一行的最小值，保持形状为50*1
    dist_1_max, _ = torch.max(dist, dim=1, keepdim=True)  # 计算每一行的最大值，保持形状为50*1
    dist = (dist - dist_1_min) / (dist_1_max - dist_1_min) + 1e-6  # 进行归一化，利用广播机制
    return dist


if __name__ == "__main__":
    a = 0
