import cv2
import numpy as np
import math
import array

class ImgProcessing():
    def __init__(self):
        pass

    @staticmethod
    def contour_to_box(contour):
        """
        Function Name: __contour_to_box

        Description: [summary]

        Argument:
            contour [] -> [sub contour from opencv]

        Return:
            [list] -> [topleft, botright]

        Edited by: [2021/1/28] [Pawat]
        """
        print(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        topleft = box.min(axis=0)
        botright = box.max(axis=0)
        return [topleft, botright]

    def box_to_center(self,box):
        """
        Function Name: box__to__center

        Description: [summary]

        Argument:
            contour [] -> [sub contour from opencv]

        Return:
            [list] -> [x center, y center]

        Edited by: [2021/4/1] [Pawat]
        """
        x0,y0 = box[0]
        x1,y1 = box[1]
        xc,yc = int((x0+x1)*0.5),int((y0+y1)*0.5)
        return [xc,yc]



    def contours_to_boxes(self, contours):
        """
        Function Name: contours_to_boxes

        Description: convert set of contours to boxes

        Argument:
            contours [[arr]] -> contours from cv2.findContours()

        Return:
            boxes[list] -> [[topleft, botright] . . . ] converted boxes from contours

        Edited by: [2021/1/28] [Pawat]
        """
        boxes = []
        for contour in contours:
            box = self.contour_to_box(contour)
            boxes.append(box)
        return boxes

    @staticmethod
    def select_closest_contour(boxes, reference,radius):  # array of boxes and center point of ref
        """
        Function Name: find_closest

        Description: [summary]

        Argument:
            boxes [list] -> [list all of boxes of detected contours]
            reference [tuplr or array] -> [reference point to compare or find cloest from boxes]

        Parameters:

        Return:
            index [int] -> [index or order of the cloest box or nothing incase there is no close box in threshlod dist]

        Edited by: [12-4-2020] [Pawat]
        """
        lengths = []
        x_lengths = []
        y_lengths = []
        index = None

        if boxes != []:

            for i, box in enumerate(boxes):
                x0, y0 = box[0]
                x1, y1 = box[1]
                center = [(x0 + x1) / 2, (y0 + y1) / 2]
                lengthX = (center[0] - reference[0]) ** 2
                lengthY = (center[1] - reference[1]) ** 2
                length = math.sqrt(lengthY + lengthX)
                x_length = math.sqrt(lengthX)
                y_length = math.sqrt(lengthY)
                lengths.append(length)
                x_lengths.append([i, x_length])
                y_lengths.append([i, y_length])
            index = None

            if lengths != []:
                lengths = np.array(lengths)
                min_length = np.min(lengths)

                if min_length <= radius:  # threshold
                    for i, length in enumerate(lengths):
                        if length == min_length:
                            index = i
                            break

        return index

    def get_x_y_from_contour(self,contour):        
        '''
        get all x coords and all y corrds from contour
        :param contour:
        :return: list of all x , list of all y coords
        '''
        x = []
        y = []
        for coord in contour:
            x.append(coord[0][0])
            y.append(coord[0][1])
        return x, y

    def area_in_bi_image(self,img):
        '''
        sum of all contour area in binary image
        :param img:
        :return: sum of areas
        '''
        # th = img
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)

        contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            areas += area
        return areas

    def find_contour_filter_area(self,binary_img,area_max,area_min = 0):
        '''

        :param binary_img:
        :param area_max:
        :param area_min:
        :return:
        '''
        if len(binary_img.shape()) == 3:
            _, th = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY_INV)
        else:
            th = binary_img

        contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = 0
        contours_new = []
        for contour in contours:

            area = cv2.contourArea(contour)
            if area < area_max and area > area_min:
                contours_new.append(contour)
            areas += area
        return contours_new, areas

    def select_biggest_contour(self, img,inv = True):
        '''
        select biggest contour
        :param img:
        :return:
        '''
        # th = img
        if inv ==True:
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
        else:
            th = img

        contours, __ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = []
        # print(len(contours))
        if contours != []:
            for contour in contours:
                area = cv2.contourArea(contour)
                areas.append(area)
            if areas != []:
                max_area = max(areas)
                for i, area in enumerate(areas):
                    if area == max_area:
                        selected_contour = [contours[i]]
            else:
                selected_contour = contours
        else:
            selected_contour = [[[[0, 0]], [[0, 0]]]]

        return selected_contour, areas

    def area_of_box(self,box):
        '''
        :param box: [toplet,botright]
        :return: area
        '''
        x0,y0 = box[0]
        x1,y1 = box[1]
        return (x1-x0)*(y1-y0)

    def area_of_box_in_yolo(self,box):
        '''
        :param box: [x0,y0,x1,y1]
        :return: area
        '''

        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

        return (x1-x0)*(y1-y0)

    def x_length_of_box_in_yolo(self,box):
        '''
        :param box: [x0,y0,x1,y1]
        :return: x_length
        '''

        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

        return x1-x0

    def y_length_of_box_in_yolo(self,box):
        '''
        :param box: [x0,y0,x1,y1]
        :return: x_length
        '''

        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

        return y1-y0