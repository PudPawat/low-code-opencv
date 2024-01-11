import cv2
import json

import numpy as np



if __name__ == '__main__':

    # Read image
    im = cv2.imread("test1/1.bmp")
    # print(im.shape)

    num_wire = 10
    sub_wire = 4

    json_wire = {}
    json_wire['main'] = {}
    json_wire["sub"] = {}
    for i in range(num_wire):
        # Select ROI
        r = cv2.selectROI(im)
        print("wire : "+str(i)+" coord: ",r)
        # Crop image
        imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        json_wire['main'][i] = r
        json_wire["sub"][i] = {}
        for j in range(sub_wire):
            # imCrop = cv2.resize(imCrop,(int(imCrop.shape[1]*2),int(imCrop.shape[0]*2)))
            sub = cv2.selectROI(imCrop)
            print("wire : " + str(i) +"  PIN : "+str(j)+ " coord: ", sub)
            # print(sub)
            json_wire["sub"][i][j] = sub



        # Display cropped image
        cv2.imshow("Image", imCrop)
        k = cv2.waitKey(0)
        if k == ord("q"):
            print(json_wire)
            with open('config/rectangles.json', 'w') as fp:
                json.dump(json_wire, fp)
            break
