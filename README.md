# PFL-WSOD
The official code of "**Proposal Feature Learning Using Proposal Relations for Weakly Supervised Object Detection**", (**IEEE ICME'24**). 

Authors: Zhaofei Wang, [Weijia Zhang](https://www.weijiazhangxh.com/), and [Min-Ling Zhang](http://palm.seu.edu.cn/zhangml/)

Results on VOC 2007 reported in our paper are obtained by using Selective Search as proposal generator in training process and using MCG/COB as proposal generator in testing process. 

Here, we report some more results. For PFL-WSOD_inter, we obtain 56.8% mAP after training on VOC 2007 trainval set (using Selective Search) and testing on VOC 2007 test set (using Selective Search), we also obtain 58.9% mAP after training on VOC 2007 trainval set (using MCG) and testing on VOC 2007 test set (using MCG).

Results for PFL-WSOD_inter on VOC 2012 reported in our paper using COB as proposal generator in testing process can be obtained in http://host.robots.ox.ac.uk:8080/anonymous/F30MQW.html.

![image](https://github.com/zhaofeiwang2000/PFL-WSOD/blob/master/network_1223.jpg)

## Get Started
### Installation
```Shell
sh install.sh
```
### Data Preparation
Download the VOC2007/VOC2012 dataset and put them into the `./data` directory. For example:
```Shell
  ./data/VOC2007/                           
  ./data/VOC2007/Annotations
  ./data/VOC2007/JPEGImages
  ./data/VOC2007/ImageSets    
```
### Training PFL-WSOD_inter
```Shell
CUDA_VISIBLE_DEVICES=0 python tools/train_net_step_inter.py --dataset voc2007 --cfg configs/PFL_voc2007.yaml --bs 1 --nw 4 --iter_size 4
```
### Testing
```Shell
CUDA_VISIBLE_DEVICES=0 python tools/test_net_inter.py --dataset voc2007test --cfg configs/PFL_voc2007.yaml --load_ckpt $model_path
```
## Acknowledgment
Our detection code is built upon [PCL](https://github.com/ppengtang/pcl.pytorch) and [NDI-WSOD](hhttps://github.com/GC-WSL/NDI). We are very grateful to all the contributors to these codebases.
