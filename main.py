import os
import cv2 as cv
from old.improc import Imageprocessing
import numpy as np
import math
import random as rng
import json
import argparse
from pypylon import pylon



def listDir(dir):
    fileNames = os.listdir(dir)
    Name = []
    for fileName in fileNames:
        Name.append(fileName)
        # print(fileName)

    return Name


def find_closest_circle(circles, reference):
    lengths = []
    index = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print(circles.shape)
        # print(circles)

        for i in circles[0, :]:
            center = (i[0], i[1])
            # print("center",center)
            lengthX = (center[0] - reference[0]) ** 2
            lengthY = (center[1] - reference[1]) ** 2
            length = math.sqrt(lengthY + lengthX)
            lengths.append(length)
            # print(center)
            # circle center
            # circle outline
            radius = i[2]
    # print(lengths)
    if lengths != []:
        lengths = np.array(lengths)
        min_length = np.min(lengths)
        for i, length in enumerate(lengths):
            if length == min_length:
                index = i

    return index

def process(frame, imgproc):
    '''
        ########## Guide line #############
    frame_sharp, params['sharp'] = imgproc.sharpen(frame)
    frame_blur, params['blur'] = imgproc.blur(frame)
    frame_binary, params['thresh'] = imgproc.threshold(frame)
    frame_line, lines, params['line'] = imgproc.line_detection(frame, frame)
    frame_HSV, params['HSV'] = imgproc.HSV_range(frame_sharp)
    frame_dialte, params['dialate'] = imgproc.dilate(frame_HSV)
    frame_erode, params['erode'] = imgproc.erode(frame_dialte)
    frame_canny, params['canny'] = imgproc.canny(frame_erode, show=True)
    canny = frame_canny.copy()
    frame_circle, circle, params['circle'] = imgproc.circle_detection(canny, frame, show=False)
        '''

    frame_result = frame.copy()

    #### working space ###
    # frame_sharp, params['sharp'] = imgproc.sharpen(frame)
    # frame_blur, params['blur'] = imgproc.blur(frame)
    # frame_binary, params['thresh'] = imgproc.threshold(frame)
    # frame_line, lines, params['line'] = imgproc.line_detection(frame, frame)
    frame_HSV, params['HSV'] = imgproc.HSV_range(frame)
    # frame_HSV = cv.resize(frame_HSV,(int(frame_HSV.shape[1]/3),int(frame_HSV.shape[0]/3)))
    frame_erode, params['erode'] = imgproc.erode(frame_HSV)
    frame_dialte, params['dilate'] = imgproc.dilate(frame_erode)

    # frame_canny, params['canny'] = imgproc.canny(frame_erode, show=True)
    # canny = frame_canny.copy()
    # frame_circle, circle, params['circle'] = imgproc.circle_detection(canny, frame, show=False)

    ### additional process ###


    return frame_result, params



def main():
    source = opt.source
    print(source)
    webcam = source.isnumeric()
    # print()
    imgproc = Imageprocessing()
    # params ={}
    global params
    params = {}
    if webcam:
        cap = cv.VideoCapture(int(source), cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    elif source == "pylon":
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        converter = pylon.ImageFormatConverter()
        # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # ========== Grabing Continusely (video) with minimal delay ==========
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # ========== converting to opencv bgr format ==========
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        rng.seed(12345)
    else:
        im_name = listDir(source)
        for name in im_name:
            key = ord("a")
            while key != ord("q"):
                frame = cv.imread(source + "/" + name)
                frame_result, params = process(frame, imgproc)
                cv.imshow("Final", frame_result)
                key = cv.waitKey(0)
                if key == ord('q') or key == 27:
                    with open('config/params.json', 'w') as fp:
                        json.dump(params, fp)
                    print(params)
                    break
    frame = None
    while True:
        # frame = cv.imread(read_dir + "/" + name)
        if webcam:
            _, frame = cap.read()
            # print(frame.shape)

        elif source == "pylon":
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                frame = image.GetArray()
        else:
            frame = cv.imread(source + "/" + name)

        if frame is None:
            break



        frame_result, params = process(frame,imgproc)
        cv.imshow("Final", frame_result)
        key = cv.waitKey(0)
        if key == ord('q') or key == 27:
            with open('config/params.json', 'w') as fp:
                json.dump(params, fp)
            print(params)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='pylon',
                        help='source pylon number for webcam')  # file/folder, 0 for webcam

    opt = parser.parse_args()

    main()
    # print(parser)
