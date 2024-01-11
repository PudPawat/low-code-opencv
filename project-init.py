import cv2
import os
import json
import numpy as np
import math
import argparse
import random as rng
import pandas as pd
from pypylon import pylon
from pathlib import Path
from lib_save import Imageprocessing, read_save
from copy import deepcopy

def listDir(dir):
    fileNames = os.listdir(dir)
    Name = []
    for fileName in fileNames:
        Name.append(fileName)
        # print(fileName)

    return Name

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
            if min_length <= 20 :
                for i, length in enumerate(lengths):
                    if length == min_length:
                        index = i
    return index

def main(params):
    source = opt.source
    # print(source)
    webcam = source.isnumeric()
    # print()
    imgproc = Imageprocessing()
    reading = read_save()
    # params ={}
    if webcam:
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    elif source == "pylon":
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        converter = pylon.ImageFormatConverter()

        # ========== Grabing Continusely (video) with minimal delay ==========
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # ========== converting to opencv bgr format ==========
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        rng.seed(12345)
    else:
        im_name = os.listdir(source) # listDir(source)
        for name in im_name:
            key = ord("a")
            while key != ord("q"):
                frame = cv2.imread(source + "/" + name)
                frame0 = frame.copy()
                img_params,_,_ = reading.read_params(params,frame)
                cv2.imshow("Final", img_params["final"]) # ori_write
                key = cv2.waitKey(0)

    frame = None
    while True:
        # frame = cv.imread(read_dir + "/" + name)

        if webcam:
            print("Open Webcam")
            _, frame = cap.read()
            # print(frame.shape)

        elif source == "pylon":
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                frame = image.GetArray()
        else:
            break

        if frame is None:
            break
        frame_ori = frame.copy()
        cv2.imshow("IMG",frame_ori)
        # frame_rect = reading.read_rectangle(rect_params, frame)
        # cv2.imshow("show",frame_rect)
        key = cv2.waitKey(1)
        if key == ord("c"):
            img_params,_,_ = reading.read_params(params, frame_ori)
            print(img_params)
            # print(frame.shape)
            # img_params_write, ori_write = process(img_params, frame_ori)
            cv2.imshow("show", img_params["final"])
            cv2.waitKey(0)

if __name__ == "__main__":
    # read params
    with Path("config/params_propeller1.json").open("r") as f:
        params = json.load(f)


    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                        help='source pylon number for webcam')
    parser.add_argument('--pylon', type=str, default='',
                        help='pylon setting(path to pylon settings)')
    # file/folder, 0 for webcam

    opt = parser.parse_args()

    # read params
    # with Path("config/params.json").open("r") as f:
    #     params = json.load(f)

    main(params)