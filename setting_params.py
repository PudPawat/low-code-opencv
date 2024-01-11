# import cv2 as cv
# from copy import deepcopy
# from old.improc import Imageprocessing
# import random as rng
# import json
# from pathlib import Path
# from easydict import EasyDict
# from pypylon import pylon
# from utils import Log, listDir
#
# class Set_params():
#
#     def __init__(self):
#         self.log = Log("main.py", stream_level="INFO", stream_enable=True, record_level="WARNING",
#                        record_path='log.csv')
#         self.log.show("===== start program =====", "INFO")
#
#         self.params = {}
#         self.rect_params = {}
#         self.config = {}
#
#         try:
#             with Path("config/main.json").open("r") as f:
#                 self.opt = json.load(f)
#                 print(self.opt)
#         except:
#             self.log.show("config/main.json" + " doesn't exist. Please, select the file again!", "ERROR")
#
#         if self.opt != {}:
#             self.opt = EasyDict(self.opt)
#             self.main()
#
#     def process(self,frame, imgproc):
#         '''
#             ########## Guide line #############
#         frame_sharp, params['sharp'] = imgproc.sharpen(frame)
#         frame_blur, params['blur'] = imgproc.blur(frame)
#         frame_binary, params['thresh'] = imgproc.threshold(frame)
#         frame_line, lines, params['line'] = imgproc.line_detection(frame, frame)
#         frame_HSV, params['HSV'] = imgproc.HSV_range(frame_sharp)
#         frame_dialte, params['dialate'] = imgproc.dilate(frame_HSV)
#         frame_erode, params['erode'] = imgproc.erode(frame_dialte)
#         frame_canny, params['canny'] = imgproc.canny(frame_erode, show=True)
#         canny = frame_canny.copy()
#         frame_circle, circle, params['circle'] = imgproc.circle_detection(canny, frame, show=False)
#             '''
#         frame_result = frame.copy()
#         for process in self.opt.basic.process:
#             if process == "sharp":
#                 frame_sharp, params['sharp'] = imgproc.sharpen(frame)
#                 frame = deepcopy(frame_sharp)
#             elif process == "blur":
#                 frame_blur, params['blur'] = imgproc.blur(frame)
#                 frame = deepcopy(frame_blur)
#             elif process == "gaussianblur":
#                 frame_blur, params['gaussianblur'] = imgproc.gaussianblur(frame)
#                 frame = deepcopy(frame_blur)
#             elif process == "thresh":
#                 frame_binary, params['thresh'] = imgproc.threshold(frame)
#                 frame = deepcopy(frame_binary)
#             elif process == "line":
#                 frame_line, lines, params['line'] = imgproc.line_detection(frame, frame)
#                 frame = deepcopy(frame_line)
#             elif process == "HSV":
#                 frame_HSV, params['HSV'] = imgproc.HSV_range(frame)
#                 frame = deepcopy(frame_HSV)
#             elif process == "dilate":
#                 frame_dialte, params['dilate'] = imgproc.dilate(frame)
#                 frame = deepcopy(frame_dialte)
#             elif process == "erode":
#                 frame_erode, params['erode'] = imgproc.erode(frame)
#                 frame = deepcopy(frame_erode)
#             elif process == "canny":
#                 frame_canny, params['canny'] = imgproc.canny(frame, show=True)
#                 frame = deepcopy(frame_canny)
#             elif process == "circle":
#                 frame_circle, circle, params['circle'] = imgproc.circle_detection(frame, frame, show=False)
#                 frame = deepcopy(frame_circle)
#             elif process == "sobel":
#                 frame_sobel, params["sobel"] = imgproc.sobel(frame)
#                 frame = deepcopy(frame_sobel)
#                 #### working space ###
#                 # frame_HSV = cv.resize(frame_HSV,(int(frame_HSV.shape[1]/3),int(frame_HSV.shape[0]/3)))
#                 # {'HSV': [0, 0, 147, 28, 47, 255], 'erode': (1, 0), 'dilate': (2, 0)}
#                 # canny = frame_canny.copy()
#             self.frame = frame
#             ### additional process ###
#
#         return frame_result, params
#
#     def main(self):
#         source = self.opt.basic.source
#         print(source)
#         webcam = source.isnumeric()
#         # print()
#         imgproc = Imageprocessing(self.opt)
#         # params ={}
#         global params
#         params = {}
#         if webcam:
#             cap = cv.VideoCapture(int(source), cv.CAP_DSHOW)
#             # cap.set(cv.CAP_PROP_FRAME_WIDTH, self.opt.basic.web_cam.width)  # add in opt
#             # cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.opt.basic.web_cam.height)  # add in opt
#             # cap.set(cv.CAP_PROP_FPS, self.opt.basic.web_cam.fps)  # add in opt
#
#         elif source == "pylon":
#             camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#             converter = pylon.ImageFormatConverter()
#             camera.Open()
#             camera.GainAuto.SetValue("Once")
#
#             camera.AcquisitionFrameRateEnable.SetValue('Off')
#             # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#
#             # ========== Grabing Continusely (video) with minimal delay ==========
#             camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
#
#             # ========== converting to opencv bgr format ==========
#             converter.OutputPixelFormat = pylon.PixelType_BGR8packed
#             converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#             rng.seed(12345)
#         else:
#             key = ord("a")
#             im_name = listDir(source)
#             for name in im_name:
#
#                 while key != ord("q"):
#                     frame = cv.imread(source + "/" + name)
#                     frame_result, params = self.process(frame, imgproc)
#                     cv.imshow("Final", frame_result)
#                     key = cv.waitKey(0)
#                     if key == ord('i'):
#                         cv.imwrite("./output/general/" +name,frame_result)
#                     if key == ord('s') :
#                         cv.imwrite("output/" + name, self.frame)
#                         with open('config/params.json', 'w') as fp:
#                             json.dump(params, fp)
#                         print(params)
#                     elif key == ord('n'):
#                         break
#                     elif key == ord('q')or key == 27:
#                         cv.destroyAllWindows()
#                         break
#
#         frame = None
#         while True:
#             # frame = cv.imread(read_dir + "/" + name)
#             if webcam:
#                 _, frame = cap.read()
#                 # print(frame.shape)
#
#             elif source == "pylon":
#                 grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#                 if grabResult.GrabSucceeded():
#                     image = converter.Convert(grabResult)
#                     frame = image.GetArray()
#             else:
#                 frame = cv.imread(source + "/" + name)
#
#             if frame is None:
#                 break
#
#             frame_result, params = self.process(frame, imgproc)
#             cv.putText(frame_result, "Press 's' to save and Press 'q' to quit",
#                        (0, frame_result.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 3)
#             cv.imshow("Final", frame_result)
#
#             key = cv.waitKey(0)
#             name = "test.jpg" ## change to date and time
#             if key == ord('i'):
#                 cv.imwrite("./output/general/"+name)
#             if key == ord('s') or key == 27:
#                 with open('config/params.json', 'w') as fp:
#                     json.dump(params, fp)
#                 print(params)
#             elif key == ord('q'):
#                 cv.destroyAllWindows()
#                 break
#
#
#
#
# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--source', type=str, default='pylon',
#     #                     help='source pylon number for webcam')  # file/folder, 0 for webcam
#     #
#     # opt = parser.parse_args()
#
#     set_param = Set_params()
#
#     # print(parser)

import os
import cv2 as cv
import cv2
from copy import deepcopy
from lib.improc import Imageprocessing
import numpy as np
import math
import random as rng
from lib.trackbar import TrackBar
import json
import argparse
from pathlib import Path
from easydict import EasyDict
from pypylon import pylon
from utils import listDir, Log, save_json, open_json


class Set_params():

    def __init__(self):
        self.log = Log("main.py", stream_level="INFO", stream_enable=True, record_level="WARNING",
                       record_path='log.csv')
        self.log.show("===== start program =====", "INFO")

        self.params = {}
        self.rect_params = {}
        self.config = {}

        try:
            with Path("../config/main.json").open("r") as f:
                self.opt = json.load(f)
                print(self.opt)
        except:
            with Path("./config/main.json").open("r") as f:
                self.opt = json.load(f)
                print(self.opt)
            self.log.show("config/main.json" + " doesn't exist. Please, select the file again!", "ERROR")

        if self.opt != {}:
            self.opt = EasyDict(self.opt)
            # self.main()

    def process(self, frame, imgproc):
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
        for process in self.opt.basic.process:
            if process == "sharp":
                frame_sharp, params['sharp'] = imgproc.sharpen(frame)
                frame = deepcopy(frame_sharp)
            elif process == "blur":
                frame_blur, params['blur'] = imgproc.blur(frame)
                frame = deepcopy(frame_blur)
            elif process == "gaussianblur":
                frame_blur, params['gaussianblur'] = imgproc.gaussianblur(frame)
                frame = deepcopy(frame_blur)
            elif process == "thresh":
                frame_binary, params['thresh'] = imgproc.threshold(frame)
                frame = deepcopy(frame_binary)
            elif process == "line":
                frame_line, lines, params['line'] = imgproc.line_detection(frame, frame)
                frame = deepcopy(frame_line)
            elif process == "HSV":
                frame_HSV, params['HSV'] = imgproc.HSV_range(frame)
                frame = deepcopy(frame_HSV)
            elif process == "dilate":
                frame_dialte, params['dilate'] = imgproc.dilate(frame)
                frame = deepcopy(frame_dialte)
            elif process == "erode":
                frame_erode, params['erode'] = imgproc.erode(frame)
                frame = deepcopy(frame_erode)
            elif process == "canny":
                frame_canny, params['canny'] = imgproc.canny(frame, show=True)
                frame = deepcopy(frame_canny)
            elif process == "circle":
                frame_circle, circle, params['circle'] = imgproc.circle_detection(frame, frame, show=False)
                frame = deepcopy(frame_circle)
            elif process == "sobel":
                frame_sobel, params["sobel"] = imgproc.sobel(frame)
                frame = deepcopy(frame_sobel)
            elif process == "barrel_distort":
                # print("barrel_distort")
                frame_barrel, params["barrel_distort"] = imgproc.barrel_distort(frame)
                # print( params["barrel_distort"])
                frame = deepcopy(frame_barrel)
            elif process == "crop":
                # print("crop")
                frame_crop, params["crop"] = imgproc.crop(frame)
                # print(params["crop"])
                frame = deepcopy(frame_crop)
            elif process == "contour_area":
                # print("contour_area")
                frame_crop, params["contour_area"] = imgproc.contour_area(frame)
                # print(params["contour_area"])
                frame = deepcopy(frame_crop)
                #### working space ###
                # frame_HSV = cv.resize(frame_HSV,(int(frame_HSV.shape[1]/3),int(frame_HSV.shape[0]/3)))
                # {'HSV': [0, 0, 147, 28, 47, 255], 'erode': (1, 0), 'dilate': (2, 0)}
                # canny = frame_canny.copy()
            self.frame = frame
            ### additional process ###

        return frame_result, params

    def main(self):
        source = self.opt.basic.source
        print(source)
        webcam = source.isnumeric()
        # print()
        imgproc = Imageprocessing(self.opt)
        # params ={}
        global params
        params = {}
        if webcam:
            cap = cv.VideoCapture(int(source), cv.CAP_DSHOW)
            # cap.set(cv.CAP_PROP_FRAME_WIDTH, self.opt.basic.web_cam.width)  # add in opt
            # cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.opt.basic.web_cam.height)  # add in opt
            # cap.set(cv.CAP_PROP_FPS, self.opt.basic.web_cam.fps)  # add in opt

        elif source == "pylon":
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            converter = pylon.ImageFormatConverter()
            camera.Open()
            camera.GainAuto.SetValue("Once")

            camera.AcquisitionFrameRateEnable.SetValue('Off')
            # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

            # ========== Grabing Continusely (video) with minimal delay ==========
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # ========== converting to opencv bgr format ==========
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            rng.seed(12345)
        else:
            key = ord("a")
            im_name = listDir(source)
            for name in im_name:

                while key != ord("q"):
                    frame = cv.imread(source + "/" + name)
                    frame_result, params = self.process(frame, imgproc)
                    cv.namedWindow("Final", cv.WINDOW_NORMAL)
                    cv.imshow("Final", frame_result)

                    key = cv.waitKey(0)
                    if key == ord('i'):
                        cv.imwrite("./output/general/" + name, frame_result)
                    if key == ord('s'):
                        cv.imwrite("output/" + name, self.frame)
                        with open('config/params.json', 'w') as fp:
                            json.dump(params, fp)
                        print(params)
                    elif key == ord('n'):
                        break
                    elif key == ord('q') or key == 27:
                        cv.destroyAllWindows()
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

            frame_result, params = self.process(frame, imgproc)
            cv.putText(frame_result, "Press 's' to save and Press 'q' to quit",
                       (0, frame_result.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 3)
            cv.imshow("Final", frame_result)

            key = cv.waitKey(0)
            name = "test.jpg"  ## change to date and time
            if key == ord('i'):
                cv.imwrite("./output/general/" + name)
            if key == ord('s') or key == 27:
                with open('config/params.json', 'w') as fp:
                    json.dump(params, fp)
                print(params)
            elif key == ord('q'):
                cv.destroyAllWindows()
                break

    def set_config(self):
        global params
        params = {}
        imgproc = Imageprocessing(self.opt)

        self.foler_dir = self.opt.basic.config_path
        self.name_format = self.opt.basic.config_name_format

        im_name = listDir(self.opt.basic.source)
        for name in im_name:
            while True:
                class_name = name.split("_")[0]
                ## json load
                config = open_json(self.foler_dir, self.name_format, class_name)

                ## read img
                frame = cv2.imread(os.path.join(self.opt.basic.source, name))
                frame_result, params = self.process(frame, imgproc)
                config["params"] = params
                key = cv.waitKey(0)
                if key == ord('s') or key == 27:
                    save_json(self.foler_dir, self.name_format, class_name, config)
                    with open('config/params.json', 'w') as fp:
                        json.dump(params, fp)
                    print(params)
                elif key == ord('n'):
                    break
                elif key == ord('q') or key == 27:
                    cv.destroyAllWindows()
                    break

            ## json save

        ## setting
        ##


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--source', type=str, default='pylon',
    #                     help='source pylon number for webcam')  # file/folder, 0 for webcam
    #
    # opt = parser.parse_args()

    set_param = Set_params()
    set_param.main()
    # set_param.set_config()

    # print(parser)

