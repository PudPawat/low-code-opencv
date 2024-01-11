import cv2
import numpy as np
import os
import math
from lib_save.read_params import *


def detect_circle(img_detect, img_draw, params = (60, 50, 0, 500)):
    rows = img_detect.shape[0]
    circles = cv2.HoughCircles(img_detect, cv2.HOUGH_GRADIENT, 1, rows / 8,
                              param1=params[0], param2=params[1],
                              minRadius=params[2], maxRadius=params[3])
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            if cal_distance_coords(center,(546.1550518881522, 421.04824877486305)) < 99999:
                # circle center
                cv2.circle(img_draw, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(img_draw, center, radius, (255, 0, 255), 3)

    return img_draw


def cal_distance_coords(point1, point2):
    '''

    :param point1: (x1 ,y1)
    :param point2: (x2 ,y2)
    :return:
    '''

    x1, y1 = point1
    x2, y2 = point2
    dis = math.sqrt((x1-x2)**2 +(y1-y2)**2)
    return dis



if __name__ == '__main__':
    # params = {'sobel': (3, 1, 1), 'gaussianblur': (1, 1), 'canny': (330, 330),'thresh':10}
    params = {'sobel': (3, 1, 1), 'canny': (300, 300),'thresh':10}
    c = {'circle': (148, 5, 0, 500)}
    circla_param = {'circle': (1, 1, 10, 500)}
    crop_circle = (546.1550518881522, 421.04824877486305, 384.2470775195958)

    path = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\light2"
    names = os.listdir(path)
    read = read_save()
    for name in names:
        img  = cv2.imread(os.path.join(path,name))
        print(img)
        img_result ,_,_ = read.read_params(params,img)

        cv2.putText(img_result["final"],str(params),(0,img_result["final"].shape[0]),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),1)
        cv2.imshow("result",img_result["final"])


        img_clrcles = detect_circle(img_result["final"], img,circla_param["circle"])


        cv2.putText(img_clrcles,str(circla_param),(0,img_clrcles.shape[0]),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),1)
        cv2.imshow("show", img_clrcles)

        cv2.waitKey(0)

        cv2.imwrite("result_{}".format(name), img_result["final"])
        cv2.imwrite("circles_{}".format(name), img_clrcles)
