# TSM：PAInet

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSM(nn.Module):
    def __init__(self, distance_type='cosine'):
        super(TSM, self).__init__()
        self.distance_type = distance_type

    def __call__(self, query, support):
        return self.forward(query, support)

    def forward(self, query, support):
        _, Ns, _ = support.size()
        dist_matrix = self.get_distance_matrix(query, support)  # (B, Nq, Ns)
        min_dist_q2s = dist_matrix.min(dim=2)[0]  # (B, Nq)
        mean_dist_q2s = min_dist_q2s.mean(dim=1)  # (B,)
        bi_mhm = mean_dist_q2s
        return bi_mhm

    def get_distance_matrix(self, query, support):
        if self.distance_type == 'cosine':
            query_norm = F.normalize(query, dim=2)
            support_norm = F.normalize(support, dim=2)
            dist_matrix = 1 - torch.bmm(query_norm, support_norm.transpose(1, 2))
            return dist_matrix
        elif self.distance_type == 'euclidean':
            query_square = query.pow(2).sum(dim=2).unsqueeze(2)
            support_square = support.pow(2).sum(dim=2).unsqueeze(1)
            dist_matrix = torch.sqrt(
                query_square + support_square - 2 * torch.bmm(query, support.transpose(1, 2)))
            return dist_matrix
        elif self.distance_type == 'manhattan':
            dist_matrix = torch.sum(torch.abs(query.unsqueeze(2) - support.unsqueeze(1)), dim=3)
            return dist_matrix
        elif self.distance_type == 'jaccard':
            query_expanded = query.unsqueeze(2)
            support_expanded = support.unsqueeze(1)
            intersection = torch.min(query_expanded, support_expanded).sum(dim=3)
            union = torch.max(query_expanded, support_expanded).sum(dim=3)
            dist_matrix = 1 - (intersection / union)
            return dist_matrix
        elif self.distance_type == 'chebyshev':
            dist_matrix = torch.max(torch.abs(query.unsqueeze(2) - support.unsqueeze(1)), dim=3)[0]  # (B, Nq, Ns)
            return dist_matrix
        else:
            raise NotImplementedError
