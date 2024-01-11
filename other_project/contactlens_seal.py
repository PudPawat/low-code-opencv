import os
import cv2
from utils.warp_polar import warp_polar
from lib_save import read_save
from copy import deepcopy


def check_seal(edge_img):
    _, contours,_ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, _ = cv2.findContours(bi_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = True
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)
        print(edge_img.shape[1])
        if w > edge_img.shape[1]/2:
            result = False
            break
        else:
            result = True
    return result

path = "../data/CONTACTLENS_SEAL/"
names = os.listdir(path)
circles = [((191, 387), 164), ((541, 399), 167), ((896, 389), 161), ((1239, 399), 167), ((179, 776), 164),
           ((538, 772), 164), ((898, 775), 157), ((1240, 774), 162)]

params = {"gaussianblur": [1, 1], "HSV": [0, 0, 100, 180, 255, 255], "erode": [1, 0], "dilate": [8, 0]}

readsave = read_save()



for name in names:
    image = cv2.imread(path + name)
    ori_img = deepcopy(image)
    cv2.imshow("ori",image)
    ## TO PROVE
    # for circle in circles:
    #     circle = [(int(circle[0][0] / 0.3), int(circle[0][1] / 0.3)), int(circle[1] / 0.3)]
    #     cv2.circle(image, circle[0], circle[1], (0, 0, 255), thickness=3)

    image_proc,_,_ = readsave.read_params(params,image)
    image = image_proc["final"]
    cv2.namedWindow("img_proc", cv2.WINDOW_NORMAL)
    cv2.imshow("img_proc", image)
    cv2.imwrite("../output/CONTACTLENS_SEAL/"+"img_proc"+name,image)

    print(name)
    # cv2.imshow("draw circle", image)
    for i, circle in enumerate(circles):
        print(i, circle[0], circle[1])
        circle = [(int(circle[0][0] / 0.3), int(circle[0][1] / 0.3)), int(circle[1] / 0.3)]
        # crop = image[circle[0][1]-circle[1]:circle[0][1]+circle[1],circle[0][0]-circle[1]:circle[0][0]+circle[1]]
        # circle = [(circle[1],circle[1]),circle[1]]
        _, warp_img = warp_polar(image, circle)
        print("warp shape", warp_img.shape)
        crop_edge = warp_img[0:warp_img.shape[0], int(warp_img.shape[1]*0.8):warp_img.shape[1]]
        result = check_seal(crop_edge)
        if result == False:
            cv2.circle(ori_img, circle[0], circle[1], (0, 0, 255), thickness=3)
        else:
            cv2.circle(ori_img, circle[0], circle[1], (0, 200, 0), thickness=3)
        cv2.imshow("warp", warp_img)
        cv2.imshow("warp1", crop_edge)
        cv2.imwrite("../output/CONTACTLENS_SEAL/" + "warp"+str(i)+name, crop_edge)
        # cv2.waitKey(0)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result",ori_img)

    cv2.imwrite("../output/CONTACTLENS_SEAL/"+"result"+name,ori_img)
    cv2.waitKey(0)
