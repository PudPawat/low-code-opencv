import cv2
import math
import os
import time
import numpy as np

def warp_polar( img_bi, circles):
    """
    warp polar : https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga49481ab24fdaa0ffa4d3e63d14c0d5e4
    :param img_bi:
    :param img_ori_result:
    :return:
    """
    warp_img = None
    if circles is not None:
        circle = circles
        # circles = sorted(circles[0],key = lambda s:s[2])
        # circle = circles[-1]
        center = (int(circle[0]), int(circle[1]))
        radius = int(circle[-1])
        dsize = (int(radius), int(2 * math.pi * radius))
        warp_img = cv2.warpPolar(img_bi, dsize, center, radius, cv2.WARP_POLAR_LINEAR)
        dst = cv2.warpPolar(warp_img, (img_bi.shape[1], img_bi.shape[0]), center, radius, flags=(cv2.WARP_INVERSE_MAP))
        # cv2.imwrite("output/reverse_original"+str(self.opt.contact_lens.area_upper)+self.name+".jpg",dst)
        ori_reverse = dst
        # if self.opt.basic.debug == "True":
        #     imshow_fit("warp_polar__reverse", dst)

    img = img_bi
    return img, warp_img

def reverse_warp(ori_image,warped_image,circles):
    '''
    TO REVERSE WARP IMAGE
    when using warp polar and want to reverse that warp image.
    :param ori_image:
    :param warped_image:x
    :param circles:
    :return:
    '''
    # ori_image = np.zeros(ori_image.shape[0:2], dtype = "unit8")
    ori_image = np.zeros(ori_image.shape, dtype=np.int8)
    # (warp_img.shape, dtype="uint8")
    if circles is not None:
        # circles = sorted(circles[0], key=lambda s: s[2])
        circle = int(circles[-1])
        center = (int(circles[0]), int(circles[1]))
        radius = circle
        # dsize = (int(radius), int(2 * math.pi * radius))
    reversed_img = cv2.warpPolar(warped_image, (ori_image.shape[1], ori_image.shape[0]), center, radius, flags=(cv2.WARP_INVERSE_MAP))
    return reversed_img


def warp_reverser_warp(img, circle = (546.1550518881522, 421.04824877486305, 375.2470775195958),crop = True):
    if crop:
        img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
    img, warp = warp_polar(img, circle)
    reversed_warp = reverse_warp(img, warp, circle)
    return reversed_warp


def preprocess(img, crop_circle):
    '''

    :param img:
    :param crop_circle:
    :return:
    '''
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
    # img_crop = img[int(crop_circle[1]/0.3):int(crop_circle[1]/0.3+crop_circle[2]/0.3),
    #            int(crop_circle[0]/0.3):int(crop_circle[0]/0.3+crop_circle[2]/0.3)]
    # cv2.imshow("resize", img)
    # img_crop = img[int(crop_circle[1] - crop_circle[2]):int(crop_circle[1] + crop_circle[2]),
    #            int(crop_circle[0] - crop_circle[2]):int(crop_circle[0] + crop_circle[2])]

    img, warp = warp_polar(img, crop_circle)
    reversed_warp = reverse_warp(img, warp, crop_circle)

    reversed_warp = reversed_warp[int(crop_circle[1] - crop_circle[2]):int(crop_circle[1] + crop_circle[2]),
                    int(crop_circle[0] - crop_circle[2]):int(crop_circle[0] + crop_circle[2])]
    return reversed_warp

if __name__ == '__main__':
    ori_image = np.zeros((100,100), dtype=np.int8)

    path = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\0927_200000"
    path = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\0924_morning"
    path = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\image\\Exposure time 200000us"
    names = os.listdir(path)

    crop_circle = (546.1550518881522, 421.04824877486305, 384.2470775195958)
    crop_circle = (546.1550518881522, 421.04824877486305, 375.2470775195958)

    for name in names:
        img = cv2.imread(os.path.join(path,name))
        img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
        # img_crop = img[int(crop_circle[1]/0.3):int(crop_circle[1]/0.3+crop_circle[2]/0.3),
        #            int(crop_circle[0]/0.3):int(crop_circle[0]/0.3+crop_circle[2]/0.3)]
        cv2.imshow("resize",img)
        img_crop = img[int(crop_circle[1]-crop_circle[2]):int(crop_circle[1]+crop_circle[2]),
                   int(crop_circle[0]-crop_circle[2]):int(crop_circle[0]+crop_circle[2])]

        img, warp = warp_polar(img,crop_circle)
        reversed_warp = reverse_warp(img,warp,crop_circle)

        reversed_warp = reversed_warp[int(crop_circle[1]-crop_circle[2]):int(crop_circle[1]+crop_circle[2]),
                        int(crop_circle[0]-crop_circle[2]):int(crop_circle[0]+crop_circle[2])]

        cv2.imwrite(os.path.join(path,"00warp_"+name), reversed_warp)
        cv2.imshow("show", reversed_warp)
        cv2.waitKey(1)
        time.sleep(0.1)