import os
import cv2
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from natsort import natsorted
from utils import get_gpus_memory_info #, calc_parameters_count
from master import classification


class mean_std(object):
    def __init__(self, img_path):
        # super(mean_std, self).__init__()

        self.img_path = img_path
        self.images = natsorted(os.listdir(self.img_path))
        # self.abnormality = abnormality

    def _read(self):
        mean = np.zeros((1, 3))
        std = np.zeros((1, 3))

        for i in range(len(self.images)):
            img = cv2.imread(os.path.join(self.img_path, self.images[i]))
            #img = img.astype(np.uint8)
            #img = cv2.resize(img, (360, 360), cv2.INTER_CUBIC)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            print(img.shape)
            img = img / 255
            m = np.mean(img, axis=(0, 1))
            s = np.std(img, axis=(0, 1))
            mean += m
            std += s
        mean = [x / i for x in mean]
        std = [x / i for x in std]
        return mean[0], std[0]

class imageloader(Dataset):

    def __init__(self, path):
        super(imageloader).__init__()

        self.path = path
        # self.mean = [0.23990076, 0.33199675, 0.50632624]
        # self.std = [0.19004004, 0.2686862, 0.37105864]
        #self.mean = mean
        #self.std = std
        self.data_info = natsorted(os.listdir(self.path))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Normalize(mean = self.mean , std = self.std)
            ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_loc = os.path.join(self.path,self.data_info[index])
        img = cv2.imread(img_loc)
        img = np.array(img)
        #img = (img-self.mean)/self.std
        #img = img.astype(np.uint8)
        #print(img.shape)
        img = cv2.resize(img, (360,360), interpolation = cv2.INTER_CUBIC)
        # cv2.imwrite(self.path+'.png', img)
        img = self.transform(img)
        return img

class Network():

    def __init__(self, whichgpu, img_path, root):
        self.whichgpu = whichgpu
        self.img_path = img_path
        self.root = root
        self._init_device()
        self._init_model()
        self._init_dataset()
        self.run()
        

    def _init_device(self):
        if torch.cuda.is_available():
            self.device_id, self.gpus_info = get_gpus_memory_info()
        else:
            print('no gpu device available')
            self.device = 'cpu'
        self.device = torch.device('cuda:{}'.format(self.whichgpu))
        cudnn.enabled = True
        cudnn.benchmark = True


    def _init_model(self):
        model = models.resnet101(pretrained=False)

        # # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
            
        model.fc = nn.Sequential(nn.Linear(2048, 1000),
                                        nn.ReLU(),
                                        nn.Linear(1000, 256),
                                          nn.ReLU(),
                                          nn.Linear(256, 1),
                                          )


        torch.cuda.set_device(self.whichgpu)
        
        self.model = model.to(self.device)
        # self.model.load_state_dict(torch.load(os.getcwd()+'/step1/weight.pth.tar')['state_dict'])
        self.model.load_state_dict(torch.load(os.getcwd()+'/step1/prevtobest_weight.pth.tar')['state_dict']) #works with pyold
        # self.model.load_state_dict(torch.load(os.getcwd()+'/step1/weight_29thbad.pth.tar')['state_dict']) #does not work with pyold
        
    def _init_dataset(self):
        #mean,std = mean_std(self.img_path)._read()
        #print('mean,std:',mean,std)
        test_images = imageloader(self.img_path)#,mean,std)
        print('No. of val images: ', len(test_images))
        self.batch_size = 1 

        self.valid_queue = data.DataLoader(test_images, batch_size=self.batch_size, num_workers=4, drop_last=False, pin_memory=True)
           

    def run(self):
        input_files = natsorted(os.listdir(self.img_path))
        root_path = self.root + '/step_1'

        k=0
        # if os.path.isdir(root_path+'/abnormal') == False:
        #     os.mkdir(root_path+'/abnormal')

        # if os.path.isdir(root_path +'/normal') == False:
        #     os.mkdir(root_path+'/normal')

        # else:
        #     print('Directory all ready exists')

        nor_path = root_path + '/Abnormal/'
        ab_path = root_path + '/Normal/'
        with torch.no_grad():
            #print('TESTING.....')
            self.model.eval()
            
            for test_img in self.valid_queue:
                test_img = test_img.to(self.device)
                t_dataload = time.time()
                y_test_pred = self.model(test_img)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_test_pred = torch.round(y_test_pred)
                y_test_pred = y_test_pred.cpu().detach().numpy()
                test_img = test_img.cpu().detach()
                #print('pred_label', y_test_pred)
                #print('imgshape', test_img.size())
                for i in range(len(y_test_pred)):
                    if y_test_pred[i] == 0.0:
                        #print('0',ab_path)
                        pred = test_img[0,:,:,:]
                        pred = pred.numpy()
                        pred = np.transpose(pred, (1,2,0))
                        pred = pred*255
                        cv2.imwrite(ab_path+input_files[k+i], pred)

                    else:
                        #print('1',nor_path)
                        pred = test_img[0,:,:,:]
                        pred = pred.numpy()
                        pred = np.transpose(pred, (1,2,0))
                        pred = pred*255
                        cv2.imwrite(nor_path+input_files[k+i], pred)
                k+=self.batch_size