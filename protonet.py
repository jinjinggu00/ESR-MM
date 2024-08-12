import torch
import torch.nn as nn
from torch.nn import functional as F
import gl
from cros_att_1 import CrossAttention as CrossAttention_1
from metric_select import metric_select
from backbone_select import Backbone


class ProtoNet(nn.Module):

    def __init__(self, opt):
        super(ProtoNet, self).__init__()
        self.model, self.out_channel, self.seq_len = Backbone(gl.dataset, gl.backbone, gl.num_chanels)

        if gl.AA == 1:
            self.attention_c = CrossAttention_1(num_attention_heads=1, input_size=self.out_channel,
                                                hidden_size=self.out_channel, hidden_dropout_prob=0.2)
        else:
            self.attention_s = None
            self.attention_c = None

    def train_mode(self, input, target, n_support):
        # input is encoder by ST_GCN
        n, c, t, v = input.size()
        '''n: 表示输入中包含的样本数
        c: channel数,表示每个样本的特征通道数
        t: time steps,表示每个样本的时间步数
        v: vertices,表示每个样本的节点数(对于图数据)'''

        def supp_idxs(cc):
            return torch.where(target.eq(cc))[0][:n_support]

        classes = torch.unique(target)
        n_class = len(classes)

        n_query = torch.where(target.eq(classes[0]))[0].size(0) - n_support

        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)

        query_idxs = torch.stack(list(map(lambda c: torch.where(target.eq(c))[0][n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]  # n是样本数, c, t, v

        if gl.AA == 0:
            z_proto = z_proto.view(n_class, n_support, c, t, v).mean(1)  # n是类数, c, t, v
        else:
            z_proto = z_proto.view(n_class, n_support, c, t, v)  # n, c, t, v
            zq, z_proto = self.hal(zq, z_proto)
            z_proto = z_proto.mean(1)

        dist, z_proto, zq = metric_select(z_proto, zq, gl.metric, n_class, n_query, t, v, c)

        # torch.Size([5, 8, 25, 256]) z_proto.shape
        # torch.Size([50, 8, 25, 256]) zq.shape
        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)
        target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
        #  dist.shape:orch.Size([50, 5])
        #  target_inds.shape:torch.Size([5, 10, 1])

        if gl.reg_rate > 0:
            if len(zq.size()) == 2:
                zq = zq.view(n_class * n_query, t, v, c)
                z_proto = z_proto.view(n_class, t, v, c)
            reg_loss = self.svd_reg_spatial(z_proto) + self.svd_reg_spatial(zq)
            rate = gl.reg_rate
            reg_loss = reg_loss * rate
            loss_val += reg_loss
        else:
            reg_loss = torch.tensor(0).float().to(gl.device)

        if gl.pca > 0:
            loss_pca = self.contrastive_loss(dist, n_class, n_query)
            loss_pca = loss_pca * gl.pca
            loss_val += loss_pca
        else:
            loss_pca = torch.tensor(0).float().to(gl.device)

        return loss_val, acc_val, reg_loss, loss_pca

    def evaluate(self, input, target, n_support):
        n, c, t, v = input.size()
        classes = torch.unique(target)
        n_class = len(classes)
        n_query = torch.where(target.eq(classes[0]))[0].size(0) - n_support

        def supp_idxs(cc):
            return torch.where(target.eq(cc))[0][:n_support]

        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)

        query_idxs = torch.stack(list(map(lambda c: torch.where(target.eq(c))[0][n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]

        if gl.AA == 0:
            z_proto = z_proto.view(n_class, n_support, c, t, v).mean(1)
        else:
            z_proto = z_proto.view(n_class, n_support, c, t, v)
            zq, z_proto = self.hal(zq, z_proto)
            z_proto = z_proto.mean(1)

        dist, z_proto, zq = metric_select(z_proto, zq, gl.metric, n_class, n_query, t, v, c)

        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)
        target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()

        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        return acc_val

    def test_ensemble(self, input, target, n_support):
        n, c, t, v = input.size()
        classes = torch.unique(target)
        n_class = len(classes)
        n_query = torch.where(target.eq(classes[0]))[0].size(0) - n_support

        def supp_idxs(cc):
            return torch.where(target.eq(cc))[0][:n_support]

        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)

        query_idxs = torch.stack(list(map(lambda c: torch.where(target.eq(c))[0][n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]

        if gl.AA == 0:
            z_proto = z_proto.view(n_class, n_support, c, t, v).mean(1)
        else:
            z_proto = z_proto.view(n_class, n_support, c, t, v)
            zq, z_proto = self.hal(zq, z_proto)
            z_proto = z_proto.mean(1)

        dist, z_proto, zq = metric_select(z_proto, zq, gl.metric, n_class, n_query, t, v, c)

        return dist

    def hal(self, zq, z_proto):
        # Get sizes of input tensors
        n, c, t, v = zq.size()
        m, h, _, _, _ = z_proto.size()

        # Reshape z_proto
        z_proto = z_proto.view(m * h, c, t, v)
        mh, _, _, _ = z_proto.size()

        # Permute and reshape zq and z_proto
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()

        # Expand zq and z_proto
        zq = torch.cat([zq, z_proto], dim=0)
        n1 = zq.size(0)
        zq = zq.unsqueeze(1).expand(n1, mh, t, v, c).reshape(n1 * mh, t, v, c)
        z_proto = z_proto.unsqueeze(0).expand(n1, mh, t, v, c).reshape(n1 * mh, t, v, c)
        nh, t, v, c = zq.size()  # nh=n1 * mh

        # Reshape zq and z_proto
        zq = zq.view(nh * t, v, c)
        z_proto = z_proto.view(nh * t, v, c)

        # Apply attention_c
        attention_c = self.attention_c(zq, z_proto)
        attention_c = attention_c.reshape(n1, mh, t, v, c).mean(1)
        attention_c = torch.squeeze(attention_c, dim=1)
        attention_c = attention_c.permute(0, 3, 1, 2)
        attention_s = attention_c[n:, :, :, :].reshape(mh, c, t, v).unsqueeze(dim=1).contiguous().reshape(m, h, c, t,
                                                                                                          v)
        attention_c = attention_c[:n, :, :, :]
        return attention_c, attention_s

    @staticmethod
    def svd_reg_spatial(x):

        if len(x.size()) == 4:
            n, t, v, c = x.size()
            x = x.view(-1, v, c)

        loss = torch.tensor(0).float().to(gl.device)
        loss -= torch.norm(torch.linalg.svdvals(x), p=2)

        return loss / x.size()[0]

    @staticmethod
    def contrastive_loss(dist, n_class, n_query):
        loss = torch.tensor(0).float().to(gl.device)
        for k in range(n_class):
            numerator = (dist[(k * n_query):((k + 1) * n_query), k]).sum(dim=0)
            denominator = (dist[[i for i in range(dist.size(0)) if
                                 i < (k * n_query) or i > ((k + 1) * n_query)], k]).sum(dim=0)
            loss += torch.log(numerator / (denominator + 1e-6))
        return loss / n_class

    def forward(self, x):
        x = self.model(x)
        return x
