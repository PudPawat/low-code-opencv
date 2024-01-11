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

    def draw_circle_one_click(self,event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            # we take note of where that mouse was located
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            drawing == True


        elif event == cv2.EVENT_LBUTTONUP:
            radius = int(math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2)))
            cv2.circle(img, (ix, iy), radius, (0, 0, 255), thickness=1)
            self.circle = ((ix, iy),radius)
            drawing = False
            return circle

    def get_coord(self,event,x,y, flags, param):
        # coords = []
        # count = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            # we take note of where that mouse was located
            ix, iy = x, y
            self.coords.append([ix, iy])
            self.count +=1
            self.coord = (x,y)
            print(self.coord)
            cv2.putText(img,str(self.coord),self.coord, cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,200),1)
            if self.count == 3:
                print("Draw circle",self.count)
                self.circle = self.define_circle(self.coords[0],self.coords[1],self.coords[2]) # return ((cx, cy), radius)
                print(self.circle, self.circle[0])
                cv2.circle(img, (int(self.circle[0][0]),int(self.circle[0][1])), int(self.circle[1]), (0, 0, 255), thickness=1)
                self.count = 0
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
    # Connects the mouse button to our callback function
    circle = None
    circles = []
    path = "F:\Ph.D\circle_classification\container-orientation-detection\dataset\\20230304\\"
    names = os.listdir(path)
    print(names)
    for name in names:
        img = cv2.imread(path + name)
        img = cv2.resize(img,(int(img.shape[1]*0.3),int(img.shape[0]*0.3)))

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw.get_coord)
        # cv2.setMouseCallback('image', draw.draw_circle_one_click)

        while (1):

            cv2.imshow('image', img)
            if circle != draw.circle:
                print("draw.circle",draw.circle)
                circle = draw.circle
                circle_resize = [(int(circle[0][0]/0.3),int(circle[0][1]/0.3)), int(circle[1]/0.3)]
                circles.append(circle_resize)

            # EXPLANATION FOR THIS LINE OF CODE:
            # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                if circles != []:
                    del circles[-1]
                print("circles",circles)
        # Once script is done, its usually good practice to call this line
        # It closes all windows (just in case you have multiple windows called)
        crop_circle = draw.circle
        # img_crop = img[int(crop_circle[1]):int(crop_circle[1] + crop_circle[2]),
        #            int(crop_circle[0]):int(crop_circle[0] + crop_circle[2])]
        print("final circle", circles)
        cv2.destroyAllWindows()