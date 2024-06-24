# 导入所需的模块和包
import torch
import torch.nn as nn
import torch.nn.functional as F



class BiMeanHausdorffMetric(nn.Module):
    def __init__(self, distance_type='cosine', temporal_consistency_weight=0.1):
        super(BiMeanHausdorffMetric, self).__init__()
        self.distance_type = distance_type
        self.temporal_consistency_weight = temporal_consistency_weight
        '''
        'temporal_consistency_weight这个参数是用于控制时间不连贯性损失的权重的。' \
        '时间不连贯性损失是指两个视频中相邻帧之间的距离变化是否平滑，' \
        '如果变化太大，说明两个视频之间的时间关系不一致，' \
        '反之则说明两个视频之间的时间关系较为一致。这个损失是用于惩罚两个视频之间的时间不连贯性的，' \
        '它与bidirectional Mean Hausdorff Metric一起构成了总损失。' \
        'temporal_consistency_weight=0.1表示时间不连贯性损失在总损失中占有10%的比重，' \
        '如果这个参数越大，表示对时间不连贯性的惩罚越重，如果这个参数越小，表示对时间不连贯性的惩罚越轻。' \
        '您可以根据您的需求来调整这个参数的值，以达到最佳的效果。'
        '''

    def __call__(self, query, support):
        return self.forward(query, support)

    def forward(self, query, support):
        # query: (B, Nq, C)  B:查询样本8原型的数量；Nq：每个样本的帧数；C：嵌入维度
        # support: (B, Ns, C) B:原型的数量；Nq：每个样本的帧数；C：嵌入维度
        B, Nq, C = query.size()
        _, Ns, _ = support.size()
        dist_matrix = self.get_distance_matrix(query, support)  # (B, Nq, Ns)
        min_dist_q2s = dist_matrix.min(dim=2)[0]  # (B, Nq)
        min_dist_s2q = dist_matrix.min(dim=1)[0]  # (B, Ns)
        mean_dist_q2s = min_dist_q2s.mean(dim=1)  # (B,)
        mean_dist_s2q = min_dist_s2q.mean(dim=1)  # (B,)
        bi_mhm = 0.5 * (mean_dist_q2s + mean_dist_s2q)  # (B,)
        temporal_consistency_loss = self.get_temporal_consistency_loss(dist_matrix)  # (B,)
        loss = bi_mhm + self.temporal_consistency_weight * temporal_consistency_loss
        return loss

    def get_distance_matrix(self, query, support):
        # query: (B, Nq, C)
        # support: (B, Ns, C)
        if self.distance_type == 'cosine':
            query_norm = F.normalize(query, dim=2)  # (B, Nq, C)
            support_norm = F.normalize(support, dim=2)  # (B, Ns, C)
            dist_matrix = 1 - torch.bmm(query_norm, support_norm.transpose(1, 2))  # (B, Nq, Ns)
            # dist_matrix = 1 - F.cosine_similarity(query.unsqueeze(2), support.unsqueeze(1), dim=3)  # (B, Nq, Ns)
            # (dist_matrix.shape):([250, 8, 8])
            return dist_matrix
        elif self.distance_type == 'euclidean':
            # dist_matrix = F.pairwise_distance(query.unsqueeze(2), support.unsqueeze(1))  # (B, Nq, Ns)
            query_square = query.pow(2).sum(dim=2).unsqueeze(2)  # (B, Nq, 1)
            support_square = support.pow(2).sum(dim=2).unsqueeze(1)  # (B, 1, Ns)
            dist_matrix = torch.sqrt(
                query_square + support_square - 2 * torch.bmm(query, support.transpose(1, 2)))  # (B, Nq, Ns)
            return dist_matrix
        elif self.distance_type == 'manhattan':
            dist_matrix = torch.sum(torch.abs(query.unsqueeze(2) - support.unsqueeze(1)), dim=3)  # (B, Nq, Ns)
            return dist_matrix
        elif self.distance_type == 'jaccard':
            query_expanded = query.unsqueeze(2)  # (B, Nq, 1, C)
            support_expanded = support.unsqueeze(1)  # (B, 1, Ns, C)
            intersection = torch.min(query_expanded, support_expanded).sum(dim=3)  # (B, Nq, Ns)
            union = torch.max(query_expanded, support_expanded).sum(dim=3)  # (B, Nq, Ns)
            dist_matrix = 1 - (intersection / union)  # (B, Nq, Ns)
            return dist_matrix
        elif self.distance_type == 'chebyshev':
            dist_matrix = torch.max(torch.abs(query.unsqueeze(2) - support.unsqueeze(1)), dim=3)[0]  # (B, Nq, Ns)
            return dist_matrix
        else:
            raise NotImplementedError

    def get_temporal_consistency_loss(self, dist_matrix):
        # dist_matrix: (B, Nq, Ns)
        B, Nq, Ns = dist_matrix.size()
        dist_matrix_shift_left = torch.cat([dist_matrix[:, 0:1], dist_matrix[:, :-1]], dim=1)  # (B, Nq-1+1=Nq ,Ns)
        dist_matrix_shift_right = torch.cat([dist_matrix[:, 1:], dist_matrix[:, -1:]], dim=1)  # (B,Nq-1+1=Nq,Ns)
        temporal_consistency_loss_left = F.relu(dist_matrix - dist_matrix_shift_left).mean(dim=(1, 2))  # (B,)
        temporal_consistency_loss_right = F.relu(dist_matrix - dist_matrix_shift_right).mean(dim=(1, 2))  # (B,)
        temporal_consistency_loss = temporal_consistency_loss_left + temporal_consistency_loss_right
        return temporal_consistency_loss
