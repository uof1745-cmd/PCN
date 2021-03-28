import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

def get_coordinate(shape, det_uv):
    b, _, w, h = shape
    uv_d = np.zeros([w, h, 2], np.float32)

    for i in range(0, w):
        for j in range(0, h):
            uv_d[i, j, 0] = j
            uv_d[i, j, 1] = i

    uv_d = np.expand_dims(uv_d.swapaxes(2, 1).swapaxes(1, 0), 0)     # 1 2 w h
    uv_d = torch.from_numpy(uv_d).cuda()
    uv_d = uv_d.repeat(b, 1, 1, 1)

    det_uv = uv_d + det_uv                                            # b 2 w h

    return det_uv

def uniform(shape, fish_uv):
    b, _, w, h = shape
    x0 = (w - 1) / 2. 

    fish_nor = (fish_uv - x0)/x0             # b 2 w h
    fish_nor = fish_nor.permute(0, 2, 3, 1)  # b w h 2
    return fish_nor

def resample_image(feature, flow):
    fish_uv = get_coordinate(feature.shape, flow)
    grid = uniform(feature.shape, fish_uv)
    target_image = F.grid_sample(feature, grid)
    return target_image
