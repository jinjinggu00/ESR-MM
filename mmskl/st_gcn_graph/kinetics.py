# 它创建了一个类的实例，打印了它的A_binary属性，并使用matplotlib.pyplot绘制了它。
# 实现了创建图的邻接矩阵并进行可视化的功能
import sys

sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from . import tools

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

num_node = 18
# 定义了节点数目为18
self_link = [(i, i) for i in range(num_node)]
# 定义了各节点自连接的边
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
outward = [(j, i) for (i, j) in inward]
# 定义了有向边,outward是其反向边
neighbor = inward + outward


# neighbor将inward和outward合并,得到全部边
# 在初始化时计算两个邻接矩阵,一个包含自连接,一个不包含自连接
class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt

    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
