from mmskl.st_gcn.st_gcn_aaai18 import ST_GCN_18
from mmskl.AGCN.agcn import AGCN_Model
from mmskl.ms_g3d.msg3d import MS_G3D_Model
from mmskl.CTRGCN.ctrgcn import CTRGCN
from mmskl.HDGCN.HDGCN import Model as HDGCN
from mmskl.stgcnpp.stgcn import STGCN


def Backbone(dataset, backbone):
    if 'ntu' in dataset or dataset == 'pku':
        node = 25
        ms_graph = 'mmskl.ms_g3d_graph.ntu_rgb_d.AdjMatrixGraph'
        sh_grpah = 'AGCN_graph.ntu_rgb_d.Graph'
        st_graph = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        ctr_graph = 'mmskl.CTRGCN_graph.ntu_rgb_d.Graph'
        hd_graph = 'mmskl.HDGCN_grpah.ntu_rgb_d_hierarchy.Graph'
        stpp_graph = {'layout': 'nturgb+d', 'mode': 'spatial'}
        seq_len = 8
    elif dataset == 'kinetics':
        node = 18
        ms_graph = 'mmskl.ms_g3d_graph.kinetics.AdjMatrixGraph'
        sh_grpah = 'AGCN_graph.kinetics.Graph'
        st_graph = {'layout': 'openpose', 'strategy': 'spatial'}
        ctr_graph = 'mmskl.CTRGCN_graph.kinetics.Graph'
        hd_graph = None
        stpp_graph = {'layout': 'openpose', 'mode': 'spatial'}
        seq_len = 13
    else:
        raise ValueError('Unknown dataset')

    if backbone == 'stgcn':
        model = ST_GCN_18(
            in_channels=3,
            num_class=60,
            dropout=0.1,
            edge_importance_weighting=False,
            graph_cfg=st_graph
        )
        out_channel = 256
    elif backbone == '2s_AGCN':
        # 使用2s-AGCN
        model = AGCN_Model(
            num_class=60,
            num_point=node,
            num_person=2,
            graph=sh_grpah,
            graph_args={'labeling_mode': 'spatial'}
        )
        out_channel = 256
    elif backbone == 'ms_g3d':
        model = MS_G3D_Model(
            num_class=60,
            num_point=node,
            num_person=2,
            num_gcn_scales=13,
            num_g3d_scales=6,
            graph=ms_graph,
        )
        out_channel = 192
    elif backbone == 'ctrgcn':
        # 使用ctrgcn
        model = CTRGCN(
            num_class=60,
            num_point=node,
            num_person=2,
            graph=ctr_graph,
            graph_args={'labeling_mode': 'spatial'}
        )
        out_channel = 256
    elif backbone == 'hdgcn':
        # 使用HDGCN
        if hd_graph is None:
            raise ValueError('HDGCN is not supported on kinetics dataset')
        model = HDGCN(
            num_class=60,
            num_point=node,
            num_person=2,
            graph=hd_graph,
            graph_args={'labeling_mode': 'spatial', 'CoM': 1}
        )
        out_channel = 256
    elif backbone == 'stgcnpp':
        # 使用stgcn++
        model = STGCN(
            graph_cfg=stpp_graph,
            in_channels=3,
            base_channels=64,
            num_person=2,
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn'
        )
        out_channel = 256
    else:
        raise ValueError('Unknown backbone')

    return model, out_channel, seq_len
