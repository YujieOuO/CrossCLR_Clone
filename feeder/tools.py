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

def reverse(data_numpy, p=0.5):

    C, T, V, M = data_numpy.shape

    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return data_numpy[:, time_range_reverse, :, :]
    else:
        return data_numpy

def motion_resample(data_numpy, max_frame=50, resample_frame=30):

    temp = torch.tensor(data_numpy)
    c, t, v, m = temp.shape
    motion = torch.zeros_like(temp)
    motion[:, :-1, :, :] = temp[:, 1:, :, :] - temp[:, :-1, :, :]
    motion = motion**2
    temporal_att = motion.mean((0,2,3))
    _,temp_list = torch.topk(temporal_att, resample_frame)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'t -> c t v m',c=c,v=v,m=m)
    output = temp.gather(1,temp_list)#[n c 30 v m]
    output = rearrange(output,'c t v m -> c (v m) t')
    output = output[:, :, None]
    output = F.interpolate(output, size=(max_frame, 1), mode='bilinear',align_corners=False)
    output = output.squeeze(dim=-1)
    output = rearrange(output,'c (v m) t -> c t v m',c=c,v=v,m=m)

    return output.numpy()