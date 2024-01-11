import cv2
import os
import numpy as np
from lib_save import  Imageprocessing, read_save

def resize_img_baseon_FOV(names):
    for name in names:
        key = ord("a")
        frame = cv2.imread(source + "/" + name)
        frame_shape = np.array(frame.shape[:2])*0.6
        frame_shape = frame_shape.astype('int32')
        print(frame_shape)
        frame = cv2.resize(frame,(frame_shape[1],frame_shape[0]))
        frame0 = frame.copy()
        img_params = reading.read_params(params, frame)
        print(img_params[0])
        img = img_params[0]["final"]
        print(img.shape)

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            if area >= 5000:
                ringt_contour = contour
                break
        M = cv2.moments(ringt_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        img_shape = img.shape # y,x
        width = 640
        height = 480
        left = int(cX - int(width/2))
        top = int(cY- height/2)

        crop_img = frame[top:top+ height, left:left+ width]
        cv2.imshow("Crop",crop_img)

        cv2.imshow("Final", img)  # ori_write
        cv2.imwrite(output_path+name,crop_img)
        key = cv2.waitKey(1)


def arrow_mask(source_result, name, params_arrow):

    frame = cv2.imread(source_result + "/" + name)
    # cv2.imshow("frame",frame)
    # print(frame)
    # print(params_arrow)
    frame_threshold = reading.read_params(params_arrow, frame)
    return frame_threshold[0]["HSV"]


if __name__ == '__main__':
    reading = read_save()
    params = {'HSV': [23, 0, 83, 109, 255, 242], 'erode': (10, 0), 'dilate': (81, 0)}
    source = "data/240degree"
    output_path = "output/originalmask/240degree/"

    im_name = os.listdir(source)
    for name in im_name:
        print(name)
        ## read originam image from stereo vision
        frame = cv2.imread(source + "/" + name)
        frame_shape_ori = frame.shape

        ## make a mask on object using params
        img_params = reading.read_params(params, frame)
        img = img_params[0]["final"]
        cv2.imwrite(output_path+name, img)
