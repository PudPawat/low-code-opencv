import cv2
from lib_save import Imageprocessing,read_save
from copy import deepcopy
import os
import math
import numpy as np

def contour_area(contours, area_min=0, area_max=1, write_area=True, draw_img = None):
    right_contours = []
    if type(draw_img) != type(None):
        draw = True
    else:
        draw = False
    for contour in contours:
        try:
            area = cv2.contourArea(contour)
            # print(area)
        except:
            continue
        if area >= area_min and area <= area_max:
            right_contours.append(contour)
            if draw:
                cv2.drawContours(draw_img, [contour], -1, (255, 0, 0), 5)
                pass
            if write_area:
                M = cv2.moments(contour)
                if M["m00"] == 0.0:
                    M["m00"] = 0.01
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(draw_img, str(area), (cX, cY), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
    return right_contours

def contour_area_by_img(bi_image, draw_img,area_min = 0,area_max =1,write_area = True):
    '''
    to filter contour by using area_min and area_max
    :param bi_image:
    :param draw_img:
    :param area_min:
    :param area_max:
    :param write_area:
    :return:
    '''
    try:
        _,contours,_ = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _,contours = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contours = contour_area(contours, area_min, area_max,write_area,draw_img)


    return right_contours, draw_img

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

def contour_center_dis(contours,ref_center, dis_criterion, draw_img = None):
    if type(draw_img) != type(None):
        draw = True
    else:
        draw = False
    right_contours = []
    for contour in contours:
        try:
            area = cv2.contourArea(contour)

        except:
            continue
        # print(area)
        M = cv2.moments(contour)
        if M["m00"] == 0.0:
            M["m00"] = 0.01
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cal_distance_coords(ref_center, (cX,cY)) < dis_criterion:
            right_contours.append(contour)
            if draw:
                cv2.drawContours(draw_img, [contour], -1, (255, 0, 0), 5)
    return right_contours

def contour_center_X_or_Y(contours,ref_center, dis_criterion,var="X", draw_img = None):
    if type(draw_img) != type(None):
        draw = True
    else:
        draw = False
    right_contours = []
    for contour in contours:
        try:
            area = cv2.contourArea(contour)
        except:
            continue
        # print(area)
        M = cv2.moments(contour)
        if M["m00"] == 0.0:
            M["m00"] = 0.01
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if var in ["X","x"]:
            print("dis",cal_distance_coords((ref_center[0],0), (cX,0)))
            if cal_distance_coords((ref_center[0],0), (cX,0)) < dis_criterion:
                right_contours.append(contour)
                if draw:
                    cv2.drawContours(draw_img, [contour], -1, (255, 0, 0), 5)

        elif var in ["Y","y"]:
            if cal_distance_coords((0,ref_center), (0,cY)) < dis_criterion:
                right_contours.append(contour)
                if draw:
                    cv2.drawContours(draw_img, [contour], -1, (255, 0, 0), 5)
    return right_contours


def contour_min_or_max(contours, mode = "max"):
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    if areas != []:
        areas_array = np.asarray(areas)
        if mode == "max":
            idx = np.argmax(areas_array)
        elif mode == "min":
            idx = np.argmin(areas_array)
        else:
            idx = np.argmax(areas_array)

        contour_max = contours[idx]
    else:
        contour_max = []
    return  contour_max


def contour_center_distance_by_img(bi_image, draw_img, ref_center, dis_criterion):
    '''
    to filter contours by using center of contour
    :param bi_image:
    :param draw_img:
    :param ref_center:
    :param dis_criterion:
    :return:
    '''
    try:
        _,contours,_ = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _,contours = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    right_contours = contour_center_dis(contours,ref_center, dis_criterion,draw_img)
    return right_contours, draw_img

def main_contour_proc(image,proc_param):
    min_area = 400
    improc = read_save()
    frame = image
    draw_frame = deepcopy(frame)
    img_params = improc.read_params(proc_param, frame)
    print(img_params[0])
    img = img_params[0]["final"]
    cv2.namedWindow("after proc",cv2.WINDOW_NORMAL)
    cv2.imshow("after proc",img)
    right_contours,draw_img = contour_area(img, draw_frame, 400 ,5000)
    # for contour in right_contours:
    #     cv2.drawContours(draw_frame)
    cv2.namedWindow("Final_contour",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Draw",cv2.WINDOW_NORMAL)
    cv2.imshow("Final_contour", img)  # ori_write
    cv2.imshow("Draw", draw_img)  # ori_write
    print("asdasd")
    if not os.path.exists("./output/casting/"+str(min_area)+"/"):
        os.mkdir("./output/casting/"+str(min_area)+"/")
    cv2.imwrite("./output/casting/"+str(min_area)+"/" + name, draw_img)
    key = cv2.waitKey(0)
    return draw_img

if __name__ == '__main__':
    source = "F:\Pawat\Projects\Imageprocessing_Vistools\data\casting"
    # params = {"gaussianblur": [5, 29], "sobel": [5, 100, 1], "blur": 14, "HSV": [0, 0, 76, 180, 255, 255], "erode": [5, 0], "dilate": [4, 0]}
    # params = {'gaussianblur': (13, 13), 'sobel': (3, 100, 6), 'blur': 16, 'HSV': [0, 0, 120, 180, 255, 255],"dilate": [2, 0], "erode": [5, 0]}
    params = {'gaussianblur': (21, 33), 'sobel': (5, 56, 1), 'blur': 1, 'HSV': [0, 0, 110, 180, 255, 220],"erode": [10, 0], "dilate": [14, 0]}
    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for name in im_name:
        frame = cv2.imread(source + "/" + name)
        main_contour_proc(frame,params)
