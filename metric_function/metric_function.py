import torch
from .BiMMH import BiMeanHausdorffMetric
from .soft_dtw import SoftDTW
from torch.nn import functional as F
from .TSM_PAINet import TSM


def dtw_loss(zq, z_proto):
    dist = dtw_dist(zq, z_proto)

    return dist


def dtw_dist(x, y, gamma=0.1):
    if len(x.size()) == 4:
        n, t, v, c = x.size()
        x = x.view(n, t, v * c)
        y = y.view(-1, t, v * c)

    n, t, c = x.size()
    m, _, _ = y.size()
    x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
    y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)
    # print(x.shape, y.shape)  ###结果([250, 8, 6400]) ([250, 8, 6400])总共要做250次序列匹配，每个序列里有8个时间步，每个时间步有6400个特征

    sdtw = SoftDTW(gamma=gamma, normalize=True)
    loss = sdtw(x, y)  # torch.Size([250])

    return loss.view(n, m)


def bimmh_dist(x, y, weight=0.1):
    if len(x.size()) == 4:
        n, t, v, c = x.size()
        x = x.view(n, t, v * c)
        y = y.view(-1, t, v * c)

    n, t, c = x.size()
    m, _, _ = y.size()
    x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
    y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)

    mhm = BiMeanHausdorffMetric(distance_type='cosine', temporal_consistency_weight=weight)
    loss = mhm(x, y)

    return loss.view(n, m)


def tsm_painet(x, y):
    if len(x.size()) == 4:
        n, t, v, c = x.size()
        x = x.view(n, t, v * c)
        y = y.view(-1, t, v * c)

    n, t, c = x.size()
    m, _, _ = y.size()
    x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
    y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)

    loss = TSM(distance_type='cosine')
    loss = loss(x, y)

    return loss.view(n, m)


def otam(x, y, method='relaxation', lbda=0.1):
    """
    Calculates the cosine similarity between the last dimension of two tensors.

    Args:
        x (torch.Tensor): Input tensor 1.
        y (torch.Tensor): Input tensor 2.
        method (str): The method to calculate OTAM. Can be 'relaxation' or 'min'. Default is 'relaxation'.
        lbda (float): The relaxation parameter. Only used when method is 'relaxation'. Default is 0.1.

    Returns:
        torch.Tensor: The OTAM distance between x and y.
    """

    n, t, v, c = x.size()
    m, _, _, _ = y.size()
    x = x.view(-1, v * c)
    y = y.view(-1, v * c)
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + 0.01
    frame_dists = torch.div(numerator, denominator)
    frame_dists = 1 - frame_dists
    frame_dists = frame_dists.reshape(n, m, t, t)
    frame_dists_1 = frame_dists.permute(0, 1, 3, 2)

    if method == 'relaxation':
        cum_dists = otam_calculate_relaxation(frame_dists, lbda) + otam_calculate_relaxation(frame_dists_1, lbda)
    elif method == 'min':
        cum_dists = otam_calculate_min(frame_dists) + otam_calculate_min(frame_dists_1)
    else:
        raise ValueError(f"Unsupported method: {method}. Method must be 'relaxation' or 'min'.")

    return cum_dists


def otam_calculate_relaxation(dists, lbda=0.1):
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] - lbda * torch.log(torch.exp(- cum_dists[:, :, 0, m - 1]))

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(
                - cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(- cum_dists[:, :, l, 0] / lbda))

        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(
                - cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(- cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


def otam_calculate_min(dists):
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] + \
                                torch.min(torch.stack([cum_dists[:, :, l - 1, 0], cum_dists[:, :, l - 1, 1],
                                                       cum_dists[:, :, l, 0]], dim=0), dim=0)[0]

        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] + torch.min(torch.stack([cum_dists[:, :, l - 1, m - 1],
                                                                               cum_dists[:, :, l, m - 1]], dim=0),
                                                                  dim=0)[0]

        cum_dists[:, :, l, -1] = dists[:, :, l, -1] + \
                                 torch.min(torch.stack([cum_dists[:, :, l - 1, -2], cum_dists[:, :, l - 1, -1],
                                                        cum_dists[:, :, l, -2]], dim=0), dim=0)[0]

    return cum_dists[:, :, -1, -1]


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
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.max(torch.abs(x - y), dim=2)[0]

    return dist


def jaccard_dist(x, y):
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


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)
