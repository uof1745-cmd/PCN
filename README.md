# Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow

## Introduction
This is a pytorch implementation for Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow.

![image](https://github.com/uof1745-cmd/PCN/blob/main/img/2.PNG)

## Requirements
* Linux or Windows
* Python 3
* Pytorch 1.5

## Dataset
For training the network,  you need to download the perspective dataset [Places2](http://places2.csail.mit.edu/download.html) or [Coco](https://cocodataset.org/). Then, move the downloaded images to
```
--data_prepare/picture
```
run
```
python data_prepare/get_dataset.py
```
to generate your fisheye dataset. The generated fisheye images and new GT will be placed in 
```
--dataset/data/train 
--dataset/gt/train  
or 
--dataset/data/test
--dataset/gt/test
```

## Training
Before training, make sure that the fisheye image and corresponding GT have been placed in 
```
--dataset/data/train
--dataset/gt/train
```
After that, generate your image lists
```
python dataset/flist.py
```
The updated file paths is in 
```
--flist/dataset/train.flist 
--flist/dataset/train_gt.flist 
```
Finally, training network by
```
python train.py
```

## Testing
If you want to use our pre-train model, you can download [here](https://pan.baidu.com/s/1_vtoyewrq6nw7t2Of-NVsw)(Extraction code: zv83) or [Google Drive](https://drive.google.com/drive/folders/1EuyGLQ7luTWimmRetA4rs79bxu9Bxm6a?usp=sharing).

Put the pre-trained model in 
```
--FISH-Net/release_model/pennet4_dataset_square256
```

Place test fisheye images and corresponding GT(not necessary, but can not be empty. You can placed the fisheye images to take up position.) in 
```
--dataset/data/test
--dataset/gt/test
```
Update file paths 
```
python dataset/flist.py
```
run
```
python test.py
```
