# Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow
This is a pytorch implementation for Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow.

## Introduction
![image](https://github.com/uof1745-cmd/PCN/blob/main/img/2.PNG)

This code contains two versions of the hyper-parameters. The first one is the implementation of node clustering task. The second one is the implementation of link prediction task.

## Requirements
* Linux or Windows
* Python 3
* Pytorch 1.5

## Dataset
For training the network,  you need to download the perspective dataset [Places2](http://places2.csail.mit.edu/download.html) or [Coco](https://cocodataset.org/). Then, move the downloaded images to
```
data_prepare\picture
```
run
```
python data_prepare/get_dataset.py
```
to generate your fisheye dataset. The generated fisheye images and new GT will be placed in 
```
dataset\data\train  or  dataset\data\test
dataset\gt\train  or  dataset\gt\test
```

## Training
Before training, make sure that the fisheye image has been placed in 
```
dataset/data/train
```

as well as corresponding GT is in 
```
dataset/gt/train
```
Update file paths in 
```
flist/dataset/train.flist 
flist/dataset/train_gt.flist 
```

run
```
python train.py
```

## Testing
If you want to use our pre-train model, you can download here.

Put the pre-train model in 
```
FISH-Net\release_model\pennet4_dataset_square256
```

placed test fisheye images in 
```
dataset/data/test
```

as well as corresponding GT is in (not necessary, but can be empty. You can placed the fisheye images to take up position.)
```
dataset/gt/test
```
Update file paths in 
```
flist/dataset/test.flist 
flist/dataset/test_gt.flist 
```

run
```
python test.py
```
