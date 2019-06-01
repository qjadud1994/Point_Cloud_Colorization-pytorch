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
(get from [Point Cloud Colorization Based on Densely Annotated 3D Shape Dataset](https://arxiv.org/ftp/arxiv/papers/1810/1810.05396.pdf) paper)

![screensh](https://github.com/qjadud1994/Point_Cloud_Colorization-pytorch/blob/master/results.PNG)

### Dataset - "DensePoint"

~~~
apt-get install zip
cd ~
mkdir DB
cd DB
mkdir densepoint
cd densepoint

export fileid=1bqaRfqcuWbvVs1YAg9vFe0zAX3jwH7zo
export filename=densepoint.zip

wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

rm -f confirm.txt cookies.txt

unzip densepoint.zip

rm densepoint.zip
~~~

### Environment

- os : Ubuntu 16.04.4 LTS
- GPU : TITAN X
- Python : 3.6.8
- Pytorch : 0.4.1
- Torchvision : 0.2.1
- tensorboardX : 1.6
