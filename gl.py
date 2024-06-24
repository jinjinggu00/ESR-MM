import numpy as np

epoch = 0
device = 'cuda:0'
experiment_root = '../output'
debug = False
local_match = 0
reg_rate = 0
threshold = 3
gamma = 0.1
iter = 0
R_ = np.random.randn(250, 15, 15)
D_ = np.random.randn(250, 15, 15)
mod = 'train'
backbone = 'st_gcn'
dataset = 'ntu120'
AA = 0
pca = 0
metric = 'eucl'
