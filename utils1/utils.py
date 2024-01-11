import os
# from .usbrelay import *
import time
import numpy as np
import cv2
import json
import pandas as pd
import copy
import math


from copy import deepcopy
from lib_save import Imageprocessing, read_save

def listDir(dir):
    """
    Function Name: listDir

    Description: Input directory and return list of all name in that directory

    Argument:
        dir [string] -> [directory]

    Return:
        [list] -> [name of all files in the directory]

    Edited by: 12-4-2020 [Pawat]
    """
    fileNames = os.listdir(dir)
    Name = []
    for fileName in fileNames:
        Name.append(fileName)
        # print(fileName)

    return Name

class Contour_area:
    def __init__(self, minarea, maxarea ):
        self.improc = read_save()

        self.minarea = minarea
        self.maxarea = maxarea

    def contour_area(self, bi_image, draw_img,area_min = 0,area_max =1,write_area = True):
        _,contours,_ = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_contours = []
        if type(draw_img) != type(None):
            draw = True
        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            if area >= area_min and area <= area_max:
                right_contours.append(contour)
                if draw:
                    cv2.drawContours(draw_img,[contour],-1,(255,0,0),5)
                    pass
                if write_area:
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(draw_img,str(area),(cX,cY),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
        return right_contours, draw_img


    def main_contour_proc(self, image,proc_param, show = True):

        frame = image
        draw_frame = deepcopy(frame)
        img_params = self.improc.read_params(proc_param, frame)
        # print(img_params[0])
        img = img_params[0]["final"]
        
        if show:
            cv2.namedWindow("after proc",cv2.WINDOW_NORMAL)
            cv2.imshow("after proc",img)
            
        right_contours,draw_img = self.contour_area(img, draw_frame, self.minarea ,self.maxarea)
        
        if show:
            cv2.namedWindow("Final_contour",cv2.WINDOW_NORMAL)
            cv2.namedWindow("Draw",cv2.WINDOW_NORMAL)
            cv2.imshow("Final_contour", img)  # ori_write
            cv2.imshow("Draw", draw_img)  # ori_write
        return draw_img



