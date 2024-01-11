
import cv2
import os
import numpy as np
from lib_save import  Imageprocessing, read_save

# def resize_img_baseon_FOV(names,source,params,output_path):
#     for name in names:
#         key = ord("a")
#         frame = cv2.imread(source + "/" + name)
#         frame_shape = np.array(frame.shape[:2])*0.6
#         frame_shape = frame_shape.astype('int32')
#         print(frame_shape)
#         frame = cv2.resize(frame,(frame_shape[1],frame_shape[0]))
#         frame0 = frame.copy()
#         img_params = reading.read_params(params, frame)
#         print(img_params[0])
#         img = img_params[0]["final"]
#         print(img.shape)
#
#         contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             print(area)
#             if area >= 5000:
#                 ringt_contour = contour
#                 break
#         M = cv2.moments(ringt_contour)
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         img_shape = img.shape # y,x
#         width = 640
#         height = 480
#         left = int(cX - int(width/2))
#         top = int(cY- height/2)
#
#         crop_img = frame[top:top+ height, left:left+ width]
#         cv2.imshow("Crop",crop_img)
#
#         cv2.imshow("Final", img)  # ori_write
#         cv2.imwrite(output_path+name,crop_img)
#         key = cv2.waitKey(1)

def arrow_mask_frame(frame, params_arrow):
    # frame = cv2.imread(source_result + "/" + name)
    # cv2.imshow("frame",frame)
    # print(frame)
    # print(params_arrow)
    frame_threshold = reading.read_params(params_arrow, frame)
    return frame_threshold[0]["final"]

def find_a_big_contour(frame,area_set = 3000):
    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area >= area_set:
            right_contour = contour
            break
    return right_contour

def crop_focus(frame, params):
    y, x, _ = (frame.shape)
    top = int(y * 0.4)  # 819
    left = int(x * 0.4)  # 1036
    print(left, top)  # 1036 819
    bot = int(y * 0.85)
    right = int(x * 0.85)
    crop_img = frame[top:bot, left:right]
    result_mask = arrow_mask_frame(crop_img, params)
    contour = find_a_big_contour(result_mask)
    crop_orchid = None
    crop_y, crop_y_h, crop_x, crop_x_w= 0,0,0,0
    if len(contour) != 0:
        x, y, w, h = cv2.boundingRect(contour)
        print("boxes", x, y, w, h)
        idx_high = np.argmax([w, h])
        aspectratio = 480 / 640
        if idx_high == 0:
            crop_orchid = crop_img[int(abs(y)):int(y + 1.4 * w * aspectratio), int(abs(x - 0.2 * w)):int(x + 1.2 * w)]
            crop_y, crop_y_h, crop_x, crop_x_w = int(abs(y)), int(y + 1.4 * w * aspectratio), int(
                abs(x - 0.2 * w)), int(x + 1.2 * w)
            print("width", crop_y, crop_y_h, crop_x, crop_x_w)

        else:
            crop_y, crop_y_h, crop_x, crop_x_w = int(abs(y - 0.1 * h)), int(y + 1.1 * h), int(abs(x)), int(
                abs(x + 1.2 * h / aspectratio))
            # crop_orchid = crop_img[int(abs(y - 0.1 * h)):int(y + 1.1 * h), int(abs(x)):int(x+1.2*h/aspectratio)]
            crop_orchid = crop_img[crop_y:crop_y_h, crop_x:crop_x_w]
            print("height", crop_y, crop_y_h, crop_x, crop_x_w)
        print(crop_orchid.shape)
        for_reverse.append([crop_y, crop_y_h, crop_x, crop_x_w])
        crop_orchid = cv2.resize(crop_orchid, (640, 480))
    return crop_orchid,[crop_y, crop_y_h, crop_x, crop_x_w]

def reverse_crop_focus(coord,crop_params, top_crop1= 819,left_crop1 = 1036):
    '''

    :param coord: (x,y)
    :param crop_params: [crop_y, crop_y_h, crop_x, crop_x_w]
    :return: new (x,y)
    '''
    new_x, new_y = (int(left_crop1 + crop_params[2] + coord[0] * (crop_params[3] - crop_params[2]) / 640),
                    int(top_crop1 + crop_params[0] + coord[1] * (crop_params[1] - crop_params[0]) / 480))
    return new_x,new_y

if __name__ == '__main__':
    reading = read_save()
    # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [5, 0], "dilate": [18, 0]}
    # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [0, 0], "dilate": [100, 0]}
    params = {'HSV': [0, 15, 0, 149, 255, 199], 'dilate': (20, 0), 'erode': (5, 0)}
    # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [5, 0], "dilate": [14, 0]}
    source = "../data/240degree"
    # source = "output/240degree" ## cropped output
    source_result = "../data/OUTPUT_PUWATs_DATA/240degree/maskrcnn_angle"
    output_path = "../output/240degree/"
    TXT_path = "../data/OUTPUT_PUWATs_DATA/240degree/TXT"

    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for_reverse = []
    for name in im_name:
        print(name)
        frame = cv2.imread(source + "/" + name)
        # factor = 0.6
        frame_shape = np.array(frame.shape[:2]) # y,x
        y,x,_ = (frame.shape)
        top = int(y*0.4) #819
        left = int(x*0.4) #1036
        print(left,top) #1036 819
        bot = int(y*0.85)
        right = int(x*0.85)
        crop_img = frame[top:bot, left:right]
        # cv2.waitKey(0)
        result_mask = arrow_mask_frame(crop_img, params)
        print("asdasd", result_mask.shape)
        contour = find_a_big_contour(result_mask)
        ## find center using moment
        if len(contour) != 0:
            # M = cv2.moments(contour)
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            print("boxes",x, y, w, h)
            # aspect_ratio.append([640- w, 480-h])
            idx_high = np.argmax([640- w, 480-h])
            idx_high = np.argmax([w, h])
            aspectratio = 3/4
            if idx_high==0:
                crop_orchid = crop_img[int(abs(y)):int(y+1.4*w*aspectratio),int(abs(x-0.2*w)):int(x+1.2*w)]
                crop_y,crop_y_h,crop_x,crop_x_w = int(abs(y)),int(y+1.4*w*aspectratio),int(abs(x-0.2*w)),int(x+1.2*w)
                print("width",crop_y, crop_y_h, crop_x, crop_x_w)

            else:
                crop_y, crop_y_h, crop_x, crop_x_w = int(abs(y - 0.1 * h)),int(y + 1.1 * h), int(abs(x)),int(abs(x+1.2*h/aspectratio))
                # crop_orchid = crop_img[int(abs(y - 0.1 * h)):int(y + 1.1 * h), int(abs(x)):int(x+1.2*h/aspectratio)]
                crop_orchid = crop_img[crop_y:crop_y_h, crop_x:crop_x_w]
                print("height",crop_y, crop_y_h, crop_x, crop_x_w)
            print(crop_orchid.shape)
            for_reverse.append([crop_y, crop_y_h, crop_x, crop_x_w])
            crop_orchid = cv2.resize(crop_orchid,(640,480))
            cv2.circle(crop_orchid,(100,100),0,(255,0,0),3)


            cv2.imshow("final", crop_orchid)
            cv2.imwrite(os.path.join(output_path,"crop"+name),crop_orchid)
            cv2.imshow("crop_img",crop_img)
            cv2.imshow("mask",result_mask)
            # cv2.waitKey(0)
            print(crop_img.shape) ##  y = 921, x = 1167
            crop_img = cv2.resize(crop_img,(640,480))
            cv2.imshow("test",crop_img)
            cv2.imwrite(output_path+name,crop_img)
            cv2.waitKey(1)


            ##### reverse back
            black_image = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
            black_image = cv2.cvtColor(black_image,cv2.COLOR_GRAY2BGR)
            print("extend",(crop_x_w-crop_x,crop_y_h-crop_y))
            extend_crop2ori = cv2.resize(crop_orchid,(crop_x_w-crop_x,crop_y_h-crop_y))
            cv2.imshow("extended",extend_crop2ori)
            black_image[top+crop_y: top+crop_y_h, left+crop_x: left+crop_x_w] = extend_crop2ori
            # cv2.circle(black_image, (int(left+crop_x+100*640/(crop_x_w-crop_x)), int(top+crop_y+100*480/(crop_y_h-crop_y))), 0, (255, 255, 0), 3)
            cv2.circle(black_image, (int(left+crop_x+100*(crop_x_w-crop_x)/640), int(top+crop_y+100*(crop_y_h-crop_y)/480)), 0, (0, 0, 0), 3)
            cv2.circle(black_image, (int(left+crop_x+100*(crop_x_w-crop_x)/640), int(top+crop_y+100*(crop_y_h-crop_y)/480)), 0, (0, 0, 255), 3)
            # cv2.circle(black_image, (int(left+crop_x+100), int(top+crop_y+100)), 0, (255, 255, 0), 3)
            cv2.imshow("reverse", black_image)
            cv2.waitKey(0)

