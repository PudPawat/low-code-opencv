import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy.spatial import distance
import time


class FeatureVisualization():
    def __init__(self, index=0, selected_layer=0, model = "alexnet"):
        self.index = index
        # self.img_path = img_path
        self.selected_layer = selected_layer

        if model == "vgg":
            # Load pretrained model
            self.pretrained_model = models.vgg16(pretrained=True)
            # print(self.pretrained_model)
            self.pretrained_model2 = models.vgg16(pretrained=True)
        elif model == "mobilenet_v2":
            self.pretrained_model = models.mobilenet_v2(pretrained=True)
            self.pretrained_model2 = models.mobilenet_v2(pretrained=True)

        elif model == "densenet121":
            self.pretrained_model = models.densenet121(pretrained=True)
            self.pretrained_model2 = models.densenet121(pretrained=True)

        elif model == "densenet201":
            self.pretrained_model = models.densenet201(pretrained=True)
            self.pretrained_model2 = models.densenet201(pretrained=True)

        elif model == "resnext50_32x4d":
            self.pretrained_model = models.resnext50_32x4d(pretrained=True)
            self.pretrained_model2 = models.resnext50_32x4d(pretrained=True)

        elif model == "vgg19":
            self.pretrained_model = models.vgg19(pretrained=True)
            self.pretrained_model2 = models.vgg19(pretrained=True)

        elif model == "alexnet":
            self.pretrained_model = models.alexnet(pretrained=True)
            self.pretrained_model2 = models.alexnet(pretrained=True)

        elif model == "squeezenet1_1":
            self.pretrained_model = models.squeezenet1_1(pretrained=True)
            self.pretrained_model2 = models.squeezenet1_1(pretrained=True)

        elif model == "mnasnet1_0":
            self.pretrained_model = models.wide_resnet101_2(pretrained=True)
            self.pretrained_model2 = models.wide_resnet101_2(pretrained=True)


        else:
            # Load pretrained model
            self.pretrained_model = models.vgg16(pretrained=True)
            # print(self.pretrained_model)
            self.pretrained_model2 = models.vgg16(pretrained=True)
            print("vgg")

        self.cuda_is_avalible = torch.cuda.is_available()
        print(self.cuda_is_avalible)
        if self.cuda_is_avalible:
            self.pretrained_model.to(torch.device("cuda:0"))
            self.pretrained_model2.to(torch.device("cuda:0"))

    # @staticmethod
    def preprocess_image(self, cv2im, resize_im=True):

        # Resize image
        if resize_im:
            cv2im = cv2.resize(cv2im, (224, 224))
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=False)
        if self.cuda_is_avalible:
            im_as_var = im_as_var.to(torch.device("cuda:0"))
        return im_as_var
    def set_index(self, index):
        self.index = index

    def process_image(self, img):
        # print('input image:')
        img = self.preprocess_image(img)
        return img

    def get_feature(self,img):
        # Image  preprocessing
        input = self.process_image(img)
        # print("input.shape:{}".format(input.shape))
        x = input
        self.pretrained_model.eval()
        with torch.no_grad():
            for index, layer in enumerate(self.pretrained_model):
                x = layer(x)
                #             print("{}:{}".format(index,x.shape))
                if (index == self.selected_layer):
                    return x

    def get_conv_feature(self,img):
        # print("1")
        # Get the feature map
        features = self.get_feature(img)
        # print("output.shape:{}".format(features.shape))
        result_path = './feat_' + str(self.selected_layer)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

    def plot_probablity(self, outputs):
        outputs = outputs.cpu()
        outputs = outputs.data.numpy()
        # print(outputs.shape)
        outputs = np.ndarray.tolist(outputs)
        # print(type(outputs),outputs)
        print(len(outputs[0]))
        x = range(0, 4096)
        # plt.bar(x, outputs[0])
        # plt.xlabel("Dimension")
        # plt.ylabel("Value")
        # plt.title("FC feature {}".format(str(self.index)))
        # plt.show()

    def get_fc_feature(self,img):
        input = self.process_image(img)
        self.pretrained_model2.eval()
        # self.pretrained_model2.classifier = nn.Sequential(*list(self.pretrained_model2.classifier.children())[0:4])
        with torch.no_grad():
            outputs = self.pretrained_model2(input)
        # self.plot_probablity(outputs)
        return outputs

    def compare_cosine(self, out1, out2):
        metric = 'cosine'
        out1 = out1.cpu()
        out2 = out2.cpu()
        cosineDistance = distance.cdist(out1, out2, metric)[0]
        # print(cosineDistance)
        # print("the distance between cat and the rocket is {}".format(cosineDistance))
        return cosineDistance

if __name__ == '__main__':
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\image\\Exposure time 60000us"
    folder_ref = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\light2_class"
    folder_ref = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\mix_focus"

    names = os.listdir(folder)
    names_ref = os.listdir(folder_ref)
    i = 0
    j = 0

    params = {'sobel': (3, 1, 1), 'gaussianblur': (1, 1), 'canny': (308, 332), 'circle': (148, 5, 0, 20)}


    def resize_scale(img, scale=0.3):
        resize = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        return resize


    featureVis = FeatureVisualization()
    all_result = []
    cv2.namedWindow("1",cv2.WINDOW_NORMAL)
    cv2.namedWindow("answer_class",cv2.WINDOW_NORMAL)
    for i in range(len(names)):
        result = []
        img1 = cv2.imread(os.path.join(folder, names[i]))
        img1 = cv2.rotate(img1,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(img1, "INPUT", (0, img1.shape[0]-10),cv2.FONT_HERSHEY_COMPLEX,2,(200,200,0),1)
        cv2.imshow("1", img1)
        # img1 = resize_scale(img1)
        imgasvar = featureVis.preprocess_image(img1)
        outputs1 = featureVis.get_fc_feature(img1)
        # print(outputs)
        featureVis.plot_probablity(outputs1)
        for j in range(len(names_ref)):
            img2 = cv2.imread(os.path.join(folder_ref, names_ref[j]))

            # cv2.imshow("2", img2)
            # img2 = resize_scale(img2)

            imgasvar = featureVis.preprocess_image(img2)
            featureVis.set_index(j)
            outputs2 = featureVis.get_fc_feature(img2)
            # print(outputs)
            featureVis.plot_probablity(outputs2)

            dis = featureVis.compare_cosine(outputs1,outputs2)
            cv2.waitKey(1)

            result.append(dis[0])

        result_array = np.asarray(result)
        ind = np.argmin(result_array)
        print("The class is {}".format(names_ref[ind]))
        answer  = cv2.imread(os.path.join(folder_ref, names_ref[ind]))

        cv2.putText(answer, "ANSWER", (0, answer.shape[0]-10),cv2.FONT_HERSHEY_COMPLEX,2,(50,0,200),1)
        cv2.imshow("answer_class", answer)
        all_result.append(result)

        cv2.waitKey(0)

    print(all_result)
    df = pd.DataFrame(all_result, columns= names_ref)
    print(df)
    df.to_csv("result.csv")
