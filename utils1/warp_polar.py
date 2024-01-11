import cv2
import math
import os

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
        center = circle[0]
        radius = circle[1]
        dsize = (int(radius), int(2 * math.pi * radius))
        warp_img = cv2.warpPolar(img_bi, dsize, center, radius, cv2.WARP_POLAR_LINEAR)
        dst = cv2.warpPolar(warp_img, (img_bi.shape[1], img_bi.shape[0]), center, radius, flags=(cv2.WARP_INVERSE_MAP))
        # cv2.imwrite("output/reverse_original"+str(self.opt.contact_lens.area_upper)+self.name+".jpg",dst)
        # self.ori_reverse = dst

    img = img_bi
    return img, warp_img

def reverse_warp():
    return True


if __name__ == '__main__':
    path = "data/CONTACTLENS_SEAL/"
    names = os.listdir(path)
    circles = [((191, 387), 164), ((541, 399), 167), ((896, 389), 161), ((1239, 399), 167), ((179, 776), 164), ((538, 772), 164), ((898, 775), 157), ((1240, 774), 162)]
    for name in names:
        image = cv2.imread(path+name)
        for circle in circles:
            circle = [(int(circle[0][0] / 0.3), int(circle[0][1] / 0.3)), int(circle[1] / 0.3)]
            cv2.circle(image, circle[0], circle[1], (0, 0, 255), thickness=3)
            
        print(name)
        cv2.imshow("draw circle",image)
        for i,circle in enumerate(circles):
            print(i,circle[0],circle[1])
            circle = [(int(circle[0][0] / 0.3), int(circle[0][1] / 0.3)), int(circle[1] / 0.3)]
            # crop = image[circle[0][1]-circle[1]:circle[0][1]+circle[1],circle[0][0]-circle[1]:circle[0][0]+circle[1]]
            # circle = [(circle[1],circle[1]),circle[1]]
            _,warp_img = warp_polar(image,circle)

            cv2.imshow("warp",warp_img)
            cv2.waitKey(0)
