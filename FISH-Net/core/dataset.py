import random
from random import shuffle
import os 
import math 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from core.utils import ZipReader


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train', level=None):
    super(Dataset, self).__init__()
    self.split = split
    self.level = level
    self.w, self.h = data_args['w'], data_args['h']
    self.data = [os.path.join(data_args['name'], i)
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'.flist'), dtype=np.str, encoding='utf-8')]
    self.gt = [os.path.join(data_args['name'], i)
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'_gt.flist'), dtype=np.str, encoding='utf-8')]
    self.data.sort()
    self.gt.sort()
    
    if split == 'train':
      temp = np.array([np.hstack((self.data)), np.hstack((self.gt))])
      temp = temp.transpose()
      shuffle(temp)
      self.data = list(temp[:, 0])
      self.gt = list(temp[:, 1])
    if debug:
      self.data = self.data[:100]
      self.gt = self.gt[:100]

  def __len__(self):
    return len(self.data)
  
  def set_subset(self, start, end):
    self.gt = self.gt[start:end]
    self.data = self.data[start:end] 

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    img_path = os.path.dirname(self.data[index])
    img_name = os.path.basename(self.data[index])
    img = Image.open(os.path.join('../', img_path, img_name)).convert('RGB')
    # load gt
    gt_path = os.path.dirname(self.gt[index])
    gt_name = os.path.basename(self.gt[index])
    gt = Image.open(os.path.join('../', gt_path, gt_name)).convert('RGB')
    img = img.resize((self.w, self.h))
    gt = gt.resize((self.w, self.h))
    return F.to_tensor(img)*2-1., F.to_tensor(gt)*2-1., img_name, gt_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item
