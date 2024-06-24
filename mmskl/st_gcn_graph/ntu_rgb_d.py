import sys

sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from . import tools

num_node = 25
# 节点数量变为25个。
self_link = [(i, i) for i in range(num_node)]
# 自连接的边
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
# 内向边
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# 将节点编号从1开始变为从0开始
outward = [(j, i) for (i, j) in inward]
# 反向边
neighbor = inward + outward


# neighbor将inward和outward合并,得到全部边

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
        # A_binary是一个二进制邻接矩阵，其中每个元素只有0或1，表示两个节点之间是否有边；
        # A_binary_with_I是一个加上单位矩阵后的邻接矩阵，其中对角线上的元素都为1，表示每个节点到自身有边；
        # A是一个对称归一化后的邻接矩阵，其中每个元素都除以了两个节点度数的乘积开方，使得每一行之和为1。


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
