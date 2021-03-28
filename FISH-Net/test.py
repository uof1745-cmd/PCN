# -*- coding: utf-8 -*-

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from PIL import Image
import math
import os
import argparse
import importlib
import datetime
import json

### My libs
from core.utils import set_device, set_seed
from core.utils import postprocess
from core.dataset import Dataset

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", default='configs/places2.json', type=str, required=False)
parser.add_argument("-l", "--level",  type=int, required=False)
parser.add_argument("-n", "--model_name", default='pennet4', type=str, required=False)
parser.add_argument("-m", "--mask", default='square', type=str)
parser.add_argument("-s", "--size", default=256, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 4

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # Model and version
  net = importlib.import_module('model.'+args.model_name)
  model = set_device(net.InpaintGenerator())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  model.load_state_dict(data['netG'])
  model.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)


  path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets

  for idx, (images, gt, img_name, gt_name) in enumerate(dataloader):
    print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      gpu, idx, len(dataloader), img_name[0]))
    images, gt = set_device([images, gt])
    with torch.no_grad():
      _, output = model(images)
    orig_imgs = postprocess(images)
    gt_imgs = postprocess(gt)
    pred_imgs = postprocess(output)
    for i in range(len(orig_imgs)):
      Image.fromarray(pred_imgs[i]).save(os.path.join(path, '{}_pred.png'.format(img_name[i].split('.')[0])))
      Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(img_name[i].split('.')[0])))
      Image.fromarray(gt_imgs[i]).save(os.path.join(path, '{}_gt.png'.format(img_name[i].split('.')[0])))

  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  config['model_name'] = args.model_name
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 
