import json
import cv2
from pathlib import Path
from lib_save.improc_save import *


imgproc = Imageprocessing()
params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [12, 0], "dilate": [14, 0]}

with Path("config/params.json").open("r") as f:
    params = json.load(f)

frame = cv2.imread("data/1.bmp")
frame0 = frame.copy()
for key in params.keys():
    print(key)
    if key == "HSV":
        # frame_HSV, params['HSV'] = imgproc.HSV_range(frame, params[key])
        frame, params['HSV'] = imgproc.HSV_range(frame, params[key])

    elif key == "erode":
        # frame_erode, params['erode'] = imgproc.erode(frame, params[key])
        frame, params['erode'] = imgproc.erode(frame, params[key])

    elif key == "dilate":
        # frame_dialte, params['dialate'] = imgproc.dilate(frame, params[key])
        frame, params['dilate'] = imgproc.dilate(frame, params[key])

    elif key == "thresh":
        # frame_binary, params['thresh'] = imgproc.threshold(frame, params[key])
        frame, params['thresh'] = imgproc.threshold(frame, params[key])

    elif key == "sharp":
        # frame_sharp, params['sharp'] = imgproc.sharpen(frame, params[key])
        frame, params['sharp'] = imgproc.sharpen(frame, params[key])

    elif key == "blur":
        # frame_blur, params['blur'] = imgproc.blur(frame, params[key])
        frame, params['blur'] = imgproc.blur(frame, params[key])

    elif key == "line":
        # frame_line, lines, params['line'] = imgproc.line_detection(frame, frame0, params[key])
        frame, lines, params['line'] = imgproc.line_detection(frame, frame0, params[key])

    elif key == "canny":
        # frame_canny, params['canny'] = imgproc.canny(frame, params[key], show=True)
        frame, params['canny'] = imgproc.canny(frame, params[key], show=True)

    elif key == "circle":
        # frame_circle, circle, params['circle'] = imgproc.circle_detection(frame, frame0, params[key], show=False)
        frame, circle, params['circle'] = imgproc.circle_detection(frame, frame0, params[key], show=False)


cv2.imshow("asd",frame)
cv2.waitKey(0)
        #
