import torch
import metric_function
from metric_function.metric_function import euclidean_dist, chebyshev_dist, jaccard_dist, manhattan_dist, cosine_dist, \
    dtw_loss, \
    bimmh_dist, otam, tsm_painet


def metric_select(z_proto, zq, metric, n_class, n_query, t, v, c):
    # 选择距离度量
    if metric == 'dtw':
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        zq = zq.permute(0, 2, 3, 1).contiguous()
        dist = dtw_loss(zq, z_proto)
    elif metric == 'eucl':
        zq = zq.reshape(n_class * n_query, -1)
        z_proto = z_proto.view(n_class, -1)
        dist = euclidean_dist(zq, z_proto)
    elif metric == 'tcmhm':
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        # print(zq.shape, z_proto.shape)#orch.Size([50, 256, 8, 25]) torch.Size([5, 256, 8, 25])
        dist = bimmh_dist(zq, z_proto)  # torch.Size([50, 5]
        # 将dist中的nan置为0
        if torch.isnan(dist).any():
            print('dist存在nan')
            # 输出nan的个数
            print(torch.count_nonzero(torch.isnan(dist)))
            dist = torch.nan_to_num(dist, nan=1e-6)
    elif metric == 'mhm':
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        # print(zq.shape, z_proto.shape)#orch.Size([50, 256, 8, 25]) torch.Size([5, 256, 8, 25])
        dist = bimmh_dist(zq, z_proto, 0)  # torch.Size([50, 5]
        if torch.isnan(dist).any():
            print('dist存在nan')
            print(torch.count_nonzero(torch.isnan(dist)))
            dist = torch.nan_to_num(dist, nan=1e-6)
    elif metric == 'tsm':
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        dist = tsm_painet(zq, z_proto)  # torch.Size([50, 5]
        if torch.isnan(dist).any():
            print('dist存在nan')
            print(torch.count_nonzero(torch.isnan(dist)))
            dist = torch.nan_to_num(dist, nan=1e-6)
    elif metric == 'otam':
        zq = zq.permute(0, 2, 3, 1).contiguous()
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        dist = otam(zq, z_proto)
    else:
        raise ValueError('Unknown metric')
    return dist, z_proto, zq
