import numpy as np
import random
import torch
from einops import rearrange, repeat
import torch.nn.functional as F

def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy

def central_spacial_mask(mask_joint):

    # 度中心性 (Degree Centrality)
    degree_centrality = [3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
                        2, 2, 2, 1, 2, 2, 2, 1, 4, 1, 2, 1, 2]
    all_joint = []
    for i in range(25):
        all_joint += [i]*degree_centrality[i]

    ignore_joint = random.sample(all_joint, mask_joint)

    return ignore_joint
    
def motion_att_temp_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## 计算 motion_attention
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    # 用abs缓解特异点影响
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

