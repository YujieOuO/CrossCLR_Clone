import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import random
from einops import rearrange, repeat


class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
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

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

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
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
    
    def central_spacial_mask(self, mask_joint):

        # 度中心性 (Degree Centrality)
        degree_centrality = [3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
                            2, 2, 2, 1, 2, 2, 2, 1, 4, 1, 2, 1, 2]
        all_joint = []
        for i in range(25):
            all_joint += [i]*degree_centrality[i]
        ignore_joint = random.sample(all_joint, mask_joint)

        return ignore_joint

    def motion_att_temp_mask(self, data, mask_frame=10):

        n, c, t, v, m = data.shape
        temp = data.clone()
        remain_num = t - mask_frame

        ## 计算 motion_attention
        motion = torch.zeros_like(temp)
        motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
        motion = -(motion)**2
        # 每一帧 c v m 维度中所有的att的和作为该帧的att
        temporal_att = motion.mean((1,3,4))

        ## 获取att最小的那些帧保留
        _,temp_list = torch.topk(temporal_att, remain_num)
        temp_list,_ = torch.sort(temp_list.squeeze())
        temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
        temp_resample = temp.gather(2,temp_list)

        ## 引入随机的temp mask
        random_frame = random.sample(range(remain_num), remain_num-mask_frame)
        random_frame.sort()
        output = temp_resample[:, :, random_frame, :, :]

        return output

    def forward(self, im_q, im_k=None, view='joint', cross=False, topk=1, context=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        if not self.pretrain:
            return self.encoder_q(im_q)
        ignore_joint = self.central_spacial_mask(mask_joint=10)

        q1 = self.encoder_q(im_q)  # queries: NxC
        q1 = F.normalize(q1, dim=1)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k1 = self.encoder_k(im_k)  # keys: NxC
            k1 = F.normalize(k1, dim=1)

        # original infonce
        l_pos = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q1, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        logit_0 = logits

        # CSM
        if random.random() < 0.5:
            im_q = self.motion_att_temp_mask(im_q)
        q2 = self.encoder_q(im_q, ignore_joint)
        q2 = F.normalize(q2, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q2, k1]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q2, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        logit_1 = logits  

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k1)
        return logit_0, logit_1, labels