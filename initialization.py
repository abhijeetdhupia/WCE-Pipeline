import numpy as np
import os
import cv2
from utils import mean_std

import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils, datasets, models
from tqdm import tqdm
from PIL import Image

from models import *

class _data(Dataset):

    """
    this class is for loading the data
    and returns only the image
    """

    def __init__(self, img_path, step):
        super(_data, self).__init__()

        self.img_path = img_path
        self.step = step

        if self.step == 1:

            self.images = datasets.ImageFolder(self.img_path)
            self._img = transforms.Compose([transforms.ToTensor])

        if self.step == 2:

            self.images = natsorted(os.listdir(img_path))
            self._img = transforms.Compose([transforms.ToTensor()])

        if self.step == 3:

            self.images = natsorted(os.listdir(self.img_path))
            # print(self.images)

            self.mean, self.std = mean_std(self.img_path, self.images)._read()
            # print('Mean: {}, Std: {}'.format(self.mean, self.std))

            self._img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # if self.step == 1:

        #     img = os.path.join(self.img_path, self.images[index])
        #     img = cv2.imread(img)
        #     img = np.array(img)
        #     img = cv2.resize(img, (100,100), interpolation = cv2.INTER_CUBIC)
        #     img = self.transform(img)

        if self.step == 2:

            img_loc = os.path.join(self.img_path, self.images[index])
            img = Image.open(img_loc).convert("RGB")
            # img = self._img
            
        if self.step == 3:

            img = cv2.imread(os.path.join(self.img_path, self.images[index]))
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        img = self._img(img)

        return img


class initial(object):
    def __init__(self, img_path, weight, step):
        super(initial, self).__init__()

        self.img_path = img_path
        self.weight_path = weight
        self.step = step

        self._init_dataset()
        self._init_device()
        self._init_model()

    def _init_device(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

    def _init_dataset(self):

        if self.step == 1:

            test_images = _data(self.img_path, self.step)
            print("No. of val images: ", len(test_images))

            self.test_queue = DataLoader(test_images, batch_size=10, num_workers=4, drop_last=False)

        if self.step == 2:
            
            test_images = _data(self.img_path, self.step)
            print("No. of val images: ", len(test_images))

            self.test_queue = DataLoader(test_images, batch_size=128, num_workers=4, drop_last=False)

        if self.step == 3:
            test_images = _data(self.img_path, self.step)

            self.batch_size = 10
            self.test_queue = DataLoader(
                test_images, batch_size=self.batch_size, num_workers=4, drop_last=False)

    def _init_model(self):

        if self.step == 1:

            model = models.resnet50(pretrained=False)

            # # Freeze parameters so we don't backprop through them
            for param in model.parameters():
                param.requires_grad = False

            model.fc = nn.Sequential(nn.Linear(2048, 1000),
                                            nn.ReLU(),
                                            nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 1),
                                            )

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(model)

            self.model = self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.weight_path)["state_dict"])

            self.model.eval()
            # print(self.model)

        if self.step == 3:

            input_files = natsorted(os.listdir(self.img_path))
            self.mean, self.std = mean_std(self.img_path, input_files)._read()
            # print('Mean: ', self.mean)
            # print('Std: ', self.std)
            self.tf = transforms.Compose(
                [transforms.Normalize((-self.mean / self.std), (1 / self.std))]
            )

            model = U_Net(img_ch=3, output_ch=1)

            if torch.cuda.device_count() > 5:
                self.model = nn.DataParallel(model)

            self.model = model.to(self.device)
            self.model.load_state_dict(
                torch.load(self.weight_path)["state_dict"])

            self.model.eval()

    def mask_color_img(self, img, mask, color=[255, 0, 255], alpha=0.5):
        """
        img: cv2 image
        mask: bool or np.where
        color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
        alpha: float [0, 1].
        """
        out = img.copy()
        img_layer = img.copy()
        img_layer[mask == 255] = color
        out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
        return out

    def visualize(self,model,data):
        data=data.to(self.device)
        outputs=model(data)
        _,preds =torch.max(outputs,1)
        return preds

    def tensor_split(self,data,pred):

        left=torch.FloatTensor().to(self.device)
        right=torch.FloatTensor().to(self.device)
        for i in range(len(pred)):
            if (pred[i] == 0):
                left=torch.cat((left, data[i].resize_(1,3,360,360)), 0)
                
            else:
                right=torch.cat((right, data[i].resize_(1,3,360,360)), 0)
        
        # print(left.shape)
        # print(right.shape)
        return left,right