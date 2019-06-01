from __future__ import print_function

import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from util import rotate_pc
import numpy as np
import cv2

class ListDataset(data.Dataset):
    def __init__(self, root, dataset, mode, num_pts, transform, augmentation=False):
        '''
        Args:
          root: (str) DB root ditectory.
          dataset : (str) Dataset name(dir).
          mode : (str) train or val or test.
          num_pts : (int) number of sampling points [2048 or 4096]
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.dataset = dataset
        self.mode = mode
        self.transform = transform
        self.num_pts = num_pts
        self.augmentation = augmentation

        self.data = []
        self.color = []

        if "densepoint" in dataset:
            self.get_DensePoint()

    def __getitem__(self, idx):

        _data = self.data[idx]
        _color = self.color[idx]

        if self.augmentation:
            _data = rotate_pc(_data)

        return _data, _color

    def __len__(self):
        return self.num_samples

    def get_DensePoint(self):
        from util import load_h5

        db_dir = os.path.join(self.root, self.dataset)

        for cat in os.listdir(db_dir):
            cat_dir = "%s/%s/PTS_%d/ply_data_%s.h5" % (db_dir, cat, self.num_pts, self.mode)

            #data, ndata, color, pid = load_h5(cat_dir, "data", "ndata", "color", "pid")
            data, color = load_h5(cat_dir, "data","color")
            self.data.append(data)
            self.color.append(color)

        self.data = np.concatenate(self.data, 0).transpose(0,2,1).astype(np.float32)
        self.color = np.concatenate(self.color, 0).transpose(0,2,1).astype(np.float32)

        self.color = self.color / 127.5 # 0~2
        self.color = self.color - 1.  # -1~1

        self.data = np.delete(self.data[0:300], trash_data_list(), 0)
        self.color = np.delete(self.color[0:300], trash_data_list(), 0)
        
        self.num_samples = self.data.shape[0]

def test1():
    
    dataset = ListDataset(root='/root/DB/', dataset='densepoint', mode="train", num_pts=2048, 
                          transform=None)

    k = dataset.data
    print(k.shape)
    
def test2():
    dataset = ListDataset(root='/root/DB/', dataset='densepoint', mode="train", num_pts=2048, 
                          transform=transforms.ToTensor(), augmentation=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for x, y in dataloader:
        print(x.shape, y.shape)
        break

# test()   
#test2()
