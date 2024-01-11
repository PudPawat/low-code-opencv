import cv2
import os
import numpy as np
from lib_save import  Imageprocessing, read_save

def resize_img_baseon_FOV(names,source,params,output_path):
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


def arrow_mask_frame(frame, params_arrow):
    # frame = cv2.imread(source_result + "/" + name)
    # cv2.imshow("frame",frame)
    # print(frame)
    # print(params_arrow)
    frame_threshold = reading.read_params(params_arrow, frame)
    return frame_threshold[0]["HSV"]

def coordinate_of_cropped_FOV(frame,params):
    ## read originam image from stereo vision
    # frame = cv2.imread(source + "/" + name)
    cv2.namedWindow("original",cv2.WINDOW_NORMAL)
    cv2.imshow("original", frame)
    # cv2.waitKey(0)
    frame_shape_ori = frame.shape
    ## reshape to crop bigger
    # factor = 0.6
    # frame_shape = np.array(frame.shape[:2]) * factor
    # frame_shape = frame_shape.astype('int32')
    # frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
    frame0 = frame.copy()

    ## make a mask on object using params
    img_params = reading.read_params(params, frame)
    img = img_params[0]["final"]
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.imshow("mask", img)
    cv2.waitKey(1)

    ## find contour and set minimum area of contour
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area >= 3000: # area criterion
            right_contour = contour
            break
    ## find center using moment
    M = cv2.moments(right_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # img_shape = img.shape  # y,x

    ## set ROI size for cropping
    width = 640
    height = 480
    left = int((cX - int(width / 2)))
    top = int((cY - height / 2))
    return left,top

if __name__ == '__main__':
    reading = read_save()
    # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [5, 0], "dilate": [18, 0]}
    # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [0, 0], "dilate": [100, 0]}
    params = {'HSV': [0, 15, 0, 149, 255, 199], 'gaussianblur': (5, 5),'treshold':(0,1), 'dilate': (5, 0), 'erode': (300, 0)}
    # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [5, 0], "dilate": [14, 0]}
    source = "../data/240degree"
    # source = "output/240degree" ## cropped output
    source_result = "../data/OUTPUT_PUWATs_DATA/240degree/maskrcnn_angle"
    output_path = "../output/240degree/"
    TXT_path = "../data/OUTPUT_PUWATs_DATA/240degree/TXT"
    params_arrow = {"HSV": [54, 52, 199, 65, 255, 255], "erode": [1, 0], "dilate": [22, 0]}

    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for name in im_name:
        print(name)
        frame = cv2.imread(source + "/" + name)
        factor = 0.6
        frame_shape = np.array(frame.shape[:2]) # y,x
        y,x,_ = (frame.shape)
        top = int(y*0.4)
        left = int(x*0.4)
        print(left,top) #1036 819
        bot = int(y*0.85)
        right = int(x*0.85)
        crop_img = frame[top:bot, left:right]
        cv2.imshow("crop_img",crop_img)
        # cv2.waitKey(0)
        result_mask = arrow_mask_frame(crop_img, params)
        print("asdasd", result_mask.shape)
        cv2.imshow("mask",result_mask)
        # cv2.waitKey(0)
        print(crop_img.shape) ##  y = 921, x = 1167
        crop_img = cv2.resize(crop_img,(640,480))
        cv2.imshow("test",crop_img)
        cv2.imwrite(output_path+name,crop_img)
        cv2.waitKey(1)


        # width = 640
        # height = 480
        # left = int(cX - int(width / 2))
        # top = int(cY - height / 2)
        try:
            ## find arrow from tang's result
            result_mask = arrow_mask(source_result,name,params_arrow)
            print("asdasd",result_mask.shape)

            ## make a mask for pasting arrow mask on to the big image
            img_black = np.zeros(frame_shape)
            img_black[top:top + height, left:left + width] = result_mask


            ## resize back to normal size
            img_black = cv2.resize(img_black,(frame_shape_ori[1],frame_shape_ori[0]))
            cv2.imshow("mask", img_black)
            cv2.imshow("Final", img)  # ori_write
            cv2.imwrite(output_path + name, img_black)
            key = cv2.waitKey(1)
        except:
            pass
