import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from tqdm import tqdm
from torchvision.utils import save_image
from initialization import initial


class _test(initial):
    def __init__(self, img_path, out_path, weight, ab=None, step=None):
        super(_test, self).__init__(img_path, weight, step)

        """
        :type img_path: string
        :param img_path: reading location of input images

        :type out_path: string
        :param out_path: saving location of prediction

        :type step: int
        :param step: current step of the program

        :raises: None

        :rtype: None
        """

        self.out_path = out_path
        self.step = step
        self.weights = weight
        self.ab = ab

    def _predict(self):

        if self.step == 1:

            k = 0
            with torch.no_grad():
                # print('TESTING.....')
                self.model.eval()

                for test_img in self.test_queue:
                    test_img = test_img.to(self.device)
                    y_test_pred = self.model(test_img)
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_test_pred = torch.round(y_test_pred)
                    y_test_pred = y_test_pred.cpu().detach().numpy()
                    test_img = test_img.cpu().detach()
                    for i in range(len(y_test_pred)):
                        if 0 in y_test_pred:
                            pred = test_img[i, :, :, :]
                            pred = pred.numpy()
                            pred = np.transpose(pred, (1, 2, 0))
                            pred = pred*255
                            pred = cv2.resize(
                                pred, (360, 360), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(self.out_path +
                                        '/Normal/' + [k+i], pred)

                        else:
                            pred = test_img[i, :, :, :]
                            pred = pred.numpy()
                            pred = np.transpose(pred, (1, 2, 0))
                            pred = pred*255
                            pred = cv2.resize(
                                pred, (360, 360), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(self.out_path +
                                        '/Abnormal/' + [k+i], pred)
                    k += self.batch_size

        if self.step == 2:

            apthae1 = torch.FloatTensor().to("cpu")
            ulcer1 = torch.FloatTensor().to("cpu")
            lymph1 = torch.FloatTensor().to("cpu")
            angio1 = torch.FloatTensor().to("cpu")
            bleed1 = torch.FloatTensor().to("cpu")
            poly1 = torch.FloatTensor().to("cpu")
            cyst1 = torch.FloatTensor().to("cpu")
            stenoses1 = torch.FloatTensor().to("cpu")
            voedemas1 = torch.FloatTensor().to("cpu")
            apthae_ulcer= torch.FloatTensor()
            bleed_angio_lymph= torch.FloatTensor()
            poly_cyst_stenoses= torch.FloatTensor()
            voedemas= torch.FloatTensor()
            apthae= torch.FloatTensor()
            ulcer= torch.FloatTensor()
            bleed= torch.FloatTensor()
            angio_lymph= torch.FloatTensor()
            poly_cyst= torch.FloatTensor()
            stenoses= torch.FloatTensor()
            lymph= torch.FloatTensor()
            angio= torch.FloatTensor()
            poly= torch.FloatTensor()
            cyst= torch.FloatTensor()


            with torch.no_grad():
                for i, inputs in enumerate(self.test_queue):
                    inputs = inputs.to(self.device)
                    model = torch.load(self.weights[0])
                    model.eval()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    four_class, five_class = self.tensor_split(inputs, preds)
                    five_class = five_class.to("cpu")
                    #print("Five class ",five_class.shape)
                    four_class = four_class.to("cpu")
                    #print("Four class ",four_class.shape)
                    torch.cuda.empty_cache()
                    del model

                    # Stage 1
                    model0 = torch.load(self.weights[1])
                    model1 = torch.load(self.weights[2])
                    model0.eval()
                    model1.eval()
                    five_class = five_class.to(self.device)
                    four_class = four_class.to(self.device)
                    if len(five_class) != 0:
                        pred_five_class = self.visualize(model0, five_class)
                        #print("five_class")
                        apthae_ulcer, bleed_angio_lymph = self.tensor_split(
                            five_class, pred_five_class)
                        
                    if len(four_class) != 0:
                        pred_four_class = self.visualize(model1, four_class)
                        #print("four_class")
                        poly_cyst_stenoses, voedemas = self.tensor_split(
                            four_class, pred_four_class)
                        
                    # print(pred_four_class)
                    del model0
                    del model1

                    # Stage 2
                    
                    
                    apthae_ulcer = apthae_ulcer.to("cpu")
                    bleed_angio_lymph = bleed_angio_lymph.to("cpu")
                    poly_cyst_stenoses = poly_cyst_stenoses.to("cpu")
                    voedemas = voedemas.to("cpu")
                    voedemas1 = torch.cat((voedemas1, voedemas), 0)
                    torch.cuda.empty_cache()

                    model00 = torch.load(self.weights[3])
                    model01 = torch.load(self.weights[4])
                    model10 = torch.load(self.weights[5])
                    model00.eval()
                    model01.eval()
                    model10.eval()
                    apthae_ulcer = apthae_ulcer.to(self.device)
                    bleed_angio_lymph = bleed_angio_lymph.to(self.device)
                    poly_cyst_stenoses = poly_cyst_stenoses.to(self.device)
                    if len(apthae_ulcer) != 0:
                        pred_apthae_ulcer = self.visualize(model00, apthae_ulcer)
                        apthae, ulcer = self.tensor_split(apthae_ulcer, pred_apthae_ulcer)
                    if len(bleed_angio_lymph) != 0:
                        pred_bleed_angio_lymph = self.visualize( model01, bleed_angio_lymph)
                        bleed, angio_lymph = self.tensor_split(bleed_angio_lymph, pred_bleed_angio_lymph)
                    if len(poly_cyst_stenoses) != 0:
                        pred_poly_cyst_stenoses = self.visualize(model10, poly_cyst_stenoses)
                        poly_cyst, stenoses = self.tensor_split(poly_cyst_stenoses, pred_poly_cyst_stenoses)
                    # print(pred_poly_cyst_stenoses)
                    del model00
                    del model01
                    del model10

                    # Stage 3
                    
                    apthae = apthae.to("cpu")
                    apthae1 = torch.cat((apthae1, apthae), 0)
                    ulcer = ulcer.to("cpu")
                    ulcer1 = torch.cat((ulcer1, ulcer), 0)
                    bleed = bleed.to("cpu")
                    bleed1 = torch.cat((bleed1, bleed), 0)
                    angio_lymph = angio_lymph.to("cpu")
                    poly_cyst = poly_cyst.to("cpu")
                    stenoses = stenoses.to("cpu")
                    stenoses1 = torch.cat((stenoses1, stenoses), 0)
                    torch.cuda.empty_cache()

                    model011 = torch.load(self.weights[6])
                    model100 = torch.load(self.weights[7])
                    model011.eval()
                    model100.eval()
                    angio_lymph = angio_lymph.to(self.device)
                    poly_cyst = poly_cyst.to(self.device)

                    if len(angio_lymph) != 0:
                        pred_angio_lymph = self.visualize(model011, angio_lymph)
                        lymph, angio = self.tensor_split(angio_lymph, pred_angio_lymph)
                    if len(poly_cyst) != 0:
                        pred_poly_cyst = self.visualize(model100, poly_cyst)
                        poly, cyst = self.tensor_split(poly_cyst, pred_poly_cyst)

                    # Stage 4
                    
                    
                    lymph = lymph.to("cpu")
                    lymph1 = torch.cat((lymph1, lymph), 0)
                    angio = angio.to("cpu")
                    angio1 = torch.cat((angio1, angio), 0)
                    poly = poly.to("cpu")
                    poly1 = torch.cat((poly1, poly), 0)
                    cyst = cyst.to("cpu")
                    cyst1 = torch.cat((cyst1, cyst), 0)
            ab_list = ["Apthae", "Ulcer", "Bleeding", "Lymphangectasias", "Angioectasias",
                    "Polypoids", "ChylousCysts", "Stenoses", "Voedemas"]
            for i in range(len(apthae1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[0], i)
                save_image(apthae1[i], name)
            for i in range(len(ulcer1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[1], i)
                save_image(ulcer1[i], name)
            for i in range(len(bleed1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[2], i)
                save_image(bleed1[i], name)
            for i in range(len(lymph1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[3], i)
                save_image(lymph1[i], name)
            for i in range(len(angio1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[4], i)
                save_image(angio1[i], name)
            for i in range(len(poly1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[5], i)
                save_image(poly1[i], name)
            for i in range(len(cyst1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[6], i)
                save_image(cyst1[i], name)
            for i in range(len(stenoses1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[7], i)
                save_image(stenoses1[i], name)
            for i in range(len(voedemas1)):
                name = self.out_path + "/{}/{}.png".format(ab_list[8], i)
                save_image(voedemas1[i], name)

        if self.step == 3:

            with torch.no_grad():
                k = 0
                for _, image in enumerate(tqdm(self.test_queue)):

                    image = image.to(self.device, dtype=torch.float32)
                    target = self.model(image)
                    target = torch.sigmoid(target)

                    for i in range(image.shape[0]):
                        img = image[i, :, :, :].detach().cpu()
                        img = self.tf(img).numpy()
                        img = np.transpose(img, (1, 2, 0))
                        img = img * 255
                        img = img.astype('uint8')

                        pred = target[i, :, :, :].detach().cpu().numpy()
                        pred = np.transpose(pred, (1, 2, 0))
                        pred = pred > 0.75
                        pred = pred * 255
                        pred.astype('uint8')
                        pred = np.squeeze(pred, axis=-1)

                        out = self.mask_color_img(img, pred)
                        out = cv2.resize(out, (360, 360), cv2.INTER_CUBIC)
                        # print('OUT_PATH: ', self.out_path)
                        name = self.out_path + '/{}.png'.format(k+i)
                        cv2.imwrite(name, out)
                        # cv2.imwrite()
                    k += self.batch_size
