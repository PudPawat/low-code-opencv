import numpy as np
import cv2
import math
import os
from bytecode import *
drawing = False  # true if mouse is pressed
ix, iy = -1, -1

# Create a function based on a CV2 Event (Left button click)
class Draw():
    def __init__(self):
        self.circle = None
        self.count = 0
        self.coords = []


    def warp_perspective(self,event,x,y, flags, param):
        # coords = []
        # count = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            # we take note of where that mouse was located
            ix, iy = x, y
            self.coords.append([ix, iy])
            self.count +=1
            self.coord = [x,y]
            cv2.putText(img,str(self.coord),self.coord, cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,200),1)
            if len(self.coords) == 4:
                ## select clock wise direction
                dst = np.array([
                    [0, 0],
                    [img.shape[1] - 1, 0],
                    [img.shape[1] - 1, img.shape[0] - 1],
                    [0, img.shape[0] - 1]], dtype="float32")
                rect = np.array(self.coords,dtype="float32")
                print(rect)
                M = cv2.getPerspectiveTransform(rect, dst)
                warp = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))
                cv2.imshow("warp",warp)
                self.coords = []


    @staticmethod
    def define_circle(p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
        return ((cx, cy), radius)
if __name__ == '__main__':

    draw = Draw()

    # Create a black image
    # img = np.zeros((512, 512, 3), np.uint8)

    # This names the window so we can reference it
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw.warp_perspective)
    # Connects the mouse button to our callback function
    circle = None
    circles = []
    path = "data/CONTACTLENS_SEAL/"
    names = os.listdir(path)
    for name in names:
        img = cv2.imread(path + name)
        img = cv2.resize(img,(int(img.shape[1]*0.3),int(img.shape[0]*0.3)))

        while (1):

            cv2.imshow('image', img)
            if circle != draw.circle:
                print(draw.circle)
                circle = draw.circle
                circle = [(int(circle[0][0]/0.3),int(circle[0][1]/0.3)), int(circle[1]/0.3)]
                circles.append(circle)

            # EXPLANATION FOR THIS LINE OF CODE:
            # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord("r"):
                if circles != []:
                    del circles[-1]
                print(circles)
        # Once script is done, its usually good practice to call this line
        # It closes all windows (just in case you have multiple windows called)
        print("final circle", circles)
        cv2.destroyAllWindows()