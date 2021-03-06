import os
import random
import subprocess

import cv2
import numpy as np
import torch
# from albumentations import (CLAHE, Compose, ElasticTransform, GridDistortion,
#                             HorizontalFlip, OneOf, OpticalDistortion,
#                             RandomBrightnessContrast, RandomGamma,
#                             RandomRotate90, ShiftScaleRotate, Transpose,
#                             VerticalFlip)
# from numba import jit
from PIL import Image
from sklearn.metrics import confusion_matrix


def metrics(pred, target, threshold=0.5):

    pred = (pred > threshold).float()
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    TN, FP, FN, TP = confusion_matrix(target, pred).ravel()

    SE = TP / (TP + FN)

    SPE = TN / (TN + FP)

    error_rate = (FP + FN) / (TP + FN + TN + FP)

    ACC = 1 - error_rate

    dice = (2 * TP) / (2 * TP + FP + FN)

    return SE, SPE, ACC, dice

class mean_std(object):

    def __init__(self, img_path, images):
        super(mean_std, self).__init__()

        self.img_path = img_path
        self.images = images

    def _read(self):
        mean = np.zeros((1, 3))
        std = np.zeros((1, 3))

        for i in range(len(self.images)):
            img = cv2.imread(os.path.join(self.img_path, self.images[i]))
            img = cv2.resize(img, (448, 448), cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            # print(img.shape)
            img = img/255
            m = np.mean(img, axis=(0,1))
            s = np.std(img, axis=(0,1))
            mean += m 
            std += s 
        mean = [x / i for x in mean]
        std = [x / i for x in std]
        return mean[0], std[0]        

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def mloss(self):
        return self.avg


# class Augmentation(object):

#     def __init__(self):
#         super(Augmentation, self).__init__()

#         self._hflip = HorizontalFlip(p=0.5)
#         self._vflip = VerticalFlip(p=0.5)
#         self._clahe = CLAHE(p=.5)
#         self._rotate = RandomRotate90(p=.5)
#         self._brightness = RandomBrightnessContrast(p=0.5)
#         self._gamma = RandomGamma(p=0.5)
#         self._transpose = Transpose(p=0.5)
#         self._elastic = ElasticTransform(
#             p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
#         self._distort = GridDistortion(p=0.5)
#         self._affine = ShiftScaleRotate(
#             shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)

#     def _aug(self):
#         iter_max = 0
#         aug = [self._hflip, self._vflip, self._clahe, self._rotate, self._brightness,
#                 self._gamma, self._transpose, self._elastic, self._distort, self._affine]
        
#         return Compose(aug)

def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""
    rst = subprocess.run('nvidia-smi -q -d Memory',stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    rst = rst.strip().split('\n')
    memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
    id = int(np.argmax(memory_available))
    return id, memory_available

def calc_parameters_count(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6
