import os
import numpy as np
import cv2
import time
from copy import deepcopy
from other_project.container.warp_and_reverse_warp import warp_polar,warp_reverser_warp
from Feature_matching_2image import matching_SIFT

path = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\image\\60000_focus"

names = os.listdir(path)

crop_circle = (546.1550518881522, 421.04824877486305, 375.2470775195958)

path_ref = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container_localize"
ref_names = os.listdir(path_ref)
ref_img = cv2.imread(os.path.join(path_ref,ref_names[0]))

for j,name in enumerate(names):

    img = cv2.imread(os.path.join(path, name))
    flag_rotate = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]
    img = cv2.rotate(img, flag_rotate[ j% 3])



    circle = (img.shape[0]/2,img.shape[1]/2,img.shape[1]/2)

    _, img_warp = warp_polar(img,circle)

    img_warp = cv2.GaussianBlur(img_warp,(3,9999),cv2.BORDER_DEFAULT)

    cv2.imshow("warp",img_warp)

    cv2.waitKey(0)

    # _, result_img = matching_SIFT(ref_img, img_warp)

    # cv2.imshow("result",result_img)
    # cv2.waitKey(0)
