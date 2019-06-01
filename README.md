# Point_Cloud_Colorization-pytorch
["Point Cloud Colorization Based on Densely Annotated 3D Shape Dataset"](https://arxiv.org/ftp/arxiv/papers/1810/1810.05396.pdf) pytorch implement

3D-point cloud colorization using GAN



### How to use

~~~
#for training
CUDA_VISIBLE_DEVICES=0 python train.py --logdir=logs/logs_temp/ --save_folder=checkpoint/checkpoint_temp/

#for testing
Inference.ipynb 
~~~

### Results
(get from [Point Cloud Colorization Based on Densely Annotated 3D Shape Dataset](https://arxiv.org/ftp/arxiv/papers/1810/1810.05396.pdf) paper

![screensh](https://github.com/qjadud1994/OCR_Detector/blob/master/photos/bad_ic15.PNG)

### Environment

- os : Ubuntu 16.04.4 LTS
- GPU : TITAN X
- Python : 3.6.8
- Pytorch : 0.4.1
- Torchvision : 0.2.1
- tensorboardX : 1.6
