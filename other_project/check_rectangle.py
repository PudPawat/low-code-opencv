import cv2
import json
import numpy as np
import math
from pathlib import Path


def find_closest(boxes, reference): #array of boxes and center point of ref
    lengths = []
    index = None
    if boxes != []:
        for i,box in enumerate(boxes):
            x0,y0 = box[0]
            x1,y1 = box[1]
            center = [(x0+x1)/2,(y0+y1)/2]
            lengthX = (center[0] - reference[0]) ** 2
            lengthY = (center[1] - reference[1]) ** 2
            length = math.sqrt(lengthY + lengthX)
            lengths.append(length)
        # print(lengths)
        if lengths != []:
            lengths = np.array(lengths)
            min_length = np.min(lengths)
            if min_length <= 8 :
                for i, length in enumerate(lengths):
                    if length == min_length:
                        index = i
    return index

with Path("config/rectangles.json").open("r") as f:
    config = json.load(f)

print(config)
print(len(config['main']))
print(len(config['sub']['0']))

img = cv2.imread("data/1.bmp")


img0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img0 = cv2.threshold(img0,0,255,cv2.THRESH_OTSU)
_,img0 = cv2.threshold(img0,0,255,cv2.THRESH_BINARY_INV)

for i in range(len(config['main'])):
    main = config['main'][str(i)]
    # imCrop = im[int(main[1]):int(main[1] + main[3]), int(main[0]):int(main[0] + main[2])]
    y0 = int(main[1])
    y1 = int(main[1] + main[3])
    x0 = int(main[0])
    x1 = int(main[0] + main[2])
    # print(type(x1))
    cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),3)

    crop = img0[y0:y1, x0:x1]

    plot = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    crop_contours, __ = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    filter_crop_contours = []
    boxes = []
    for contour in crop_contours:
        area = cv2.contourArea(contour)
        if area > 5:
            filter_crop_contours.append(contour)

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            topleft = box.min(axis=0)
            botright = box.max(axis=0)
            cv2.drawContours(plot, [box], 0, (0, 0, 255), 2)
            cv2.drawContours(plot, contour, 0, (255, 0, 0), 2)
            boxes.append([topleft,botright])
            print(area)

            # cv2.imshow("qwe", plot)
            # cv2.waitKey(0)
    print(boxes)
    print(len(boxes))

    for j in range(len(config['sub'][str(i)])):
        sub = config['sub'][str(i)][str(j)]
        sub_y0 = int(sub[1])
        sub_y1 = int(sub[1] + sub[3])
        sub_x0 = int(sub[0])
        sub_x1 = int(sub[0] + sub[2])
        cv2.rectangle(img, (x0 + sub_x0, y0 + sub_y0), (x0 + sub_x1, y0 + sub_y1), (0, 0, 255), 3) # draw in big picture
        center_ref = ((sub_x0+sub_x1)/2, (sub_y0+sub_y1)/2)
        index = find_closest(boxes,center_ref)
        if index != None:
        # print(index)
            area = cv2.contourArea(filter_crop_contours[index])
            print("area",area)
            if area > 90:
                result = "NG"
            else:
                result = "OK"
        else:
            result = "OK"
        print(result)
    cv2.imshow("asd",crop)
    cv2.waitKey(0)
