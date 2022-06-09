import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=2048, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, lambd = 5e-4,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.lambd = lambd

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(
                                nn.Linear(dim_mlp, feature_dim, bias=False),
                                nn.BatchNorm1d(feature_dim),
                                nn.ReLU(),
                                nn.Linear(feature_dim, feature_dim, bias=False),
                                nn.BatchNorm1d(feature_dim),
                                nn.ReLU(),
                                nn.Linear(feature_dim, feature_dim, bias=False),
                            )

    def forward(self, im_q, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            return self.encoder_q(im_q)

        # compute query features
        feat1 = self.encoder_q(im_q)  # queries: NxC
        feat1 = F.normalize(feat1, dim=1)

        feat2 = self.encoder_q(im_k)  # keys: NxC
        feat2 = F.normalize(feat2, dim=1)

        feat1_norm = (feat1 - feat1.mean(0)) / feat1.std(0)
        feat2_norm = (feat2 - feat2.mean(0)) / feat2.std(0)

        N, D = feat1_norm.shape
        c = feat1_norm.T @ feat2_norm
        c.div_(N)
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        print(on_diag)
        print(off_diag)

        BTloss = on_diag + self.lambd * off_diag

        return BTloss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        