import numpy as np
import math
from .trackbar import *
import cv2 as cv


class Imageprocessing(object):

    def __init__(self,opt):
        for improc_module in opt.basic.process :
            if improc_module == "sharp":
                self.var_sharpen = TrackBar.Sharpen()
            elif improc_module == "blur":
                self.var_blur = TrackBar.Blur()

            elif improc_module == "gaussianblur":
                self.var_gaussianblur = TrackBar.GaussianBlur()

            elif improc_module == "thresh":
                self.var_binary = TrackBar.Binary()

            elif improc_module == "line":
                self.var_line_det = TrackBar.LineDetection()

            elif improc_module == "HSV":
                self.var_HSV_range = TrackBar.HSV()

            elif improc_module == "dilate":
                self.var_dilate = TrackBar.Dilate()

            elif improc_module == "erode":
                self.var_erode = TrackBar.Erode()

            elif improc_module == "canny":
                self.var_canny = TrackBar.Canny()

            elif improc_module == "circle":
                self.var_circle_det = TrackBar.CircleDetection()

            elif improc_module == "sobel":
                self.var_sobel = TrackBar.Sobel()





        #### another state #####
        # self.var_canny_1 = TrackBar.Canny1()
        # self.var_circle_det_1 = TrackBar.CircleDetection1()
        # self.var_HSV_range_1 = TrackBar.HSV1()

    def threshold(self, img, show = True):
        '''
        Threshold : setting threshold value
        :param img:
        :param show:
        :return:
        '''

        th_val = self.var_binary.return_var()
        if th_val == 0 :
            flag = cv.THRESH_BINARY+cv.THRESH_OTSU

        else:
            flag = cv.THRESH_BINARY

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, th = cv.threshold(img,th_val, 255, flag)
        if show == True:
            cv.imshow(self.var_binary.window_binary_name, th)

        return th, (th_val)

    def canny(self,img, show = True):
        '''
        edge detection : there is two params X ,Y
        :param img:
        :param show:
        :return: image, (Y_val, X_val)
        '''
        Y_val, X_val = self.var_canny.return_var()

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, Y_val, X_val)
        print(canny)
        if show == True:
            cv.imshow(self.var_canny.window_canny_name, canny)
        return canny, (Y_val, X_val)

    def canny_1(self,img, show = True):
        '''
        edge detection : there is two params X ,Y
        :param img:
        :param show:
        :return: image, (Y_val, X_val)
        '''
        Y_val, X_val = self.var_canny_1.return_var()

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, Y_val, X_val)
        if show == True:
            cv.imshow(self.var_canny_1.window_canny_name, canny)
        return canny, (Y_val, X_val)

    def blur(self,img, show = True):
        '''
        Buring
        :param img:
        :param show:
        :return: blur,(filter_size)
        '''

        filter_size = self.var_blur.return_var()
        if filter_size < 1 :
            filter_size = 1

        blur = cv.blur(img, (int(filter_size), int(filter_size)))

        if show == True:
            cv.imshow(self.var_blur.window_blur_name, blur)

        return blur,(filter_size)

    def gaussianblur(self,img, show = True):
        '''
        Buring
        :param img:
        :param show:
        :return: blur,(filter_size)
        '''

        x,y = self.var_gaussianblur.return_var()

        if not (x > 0 and x % 2 == 1):
            x = x+1
        if not (y > 0 and y % 2 == 1):
            y =y +1
        blur = cv.GaussianBlur(img, (int(x), int(y)),0)

        if show == True:
            cv.imshow(self.var_gaussianblur.window_blur_name, blur)

        return blur,(x,y)

    def HSV_range(self,img, mode = "HSV",show = True):
        '''
        Thresholding by HSV : by setting lower bound and upper bound
        :param img:
        :param mode:
        :param show:
        :return: Image ,(low_H, low_S, low_V, high_H, high_S, high_V)
        '''
        low_H, low_S, low_V, high_H, high_S, high_V = self.var_HSV_range.return_var()

        print(len(img.shape))
        if len(img.shape) != 3:
            img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        else:

            if mode == "HSV":
                frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            elif mode == "HLS":
                frame_HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
                frame_threshold = cv.inRange(frame_HLS, (low_H, low_V, low_S), (high_H, high_V, high_S))


        if show == True:
            cv.imshow(self.var_HSV_range.window_detection_name, frame_threshold)

        return  frame_threshold, [low_H, low_S, low_V, high_H, high_S, high_V]

    def HSV_range_1(self,img, mode = "HSV",show = True):
        '''
        Thresholding by HSV : by setting lower bound and upper bound
        :param img:
        :param mode:
        :param show:
        :return: Image ,(low_H, low_S, low_V, high_H, high_S, high_V)
        '''
        low_H, low_S, low_V, high_H, high_S, high_V = self.var_HSV_range_1.return_var()

        if mode == "HSV":
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # frame_HSV = cv.cvtColor(frame_HSV, cv.COLOR_HSV2BGR)

            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif mode == "HLS":
            frame_HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            frame_threshold = cv.inRange(frame_HLS, (low_H, low_V, low_S), (high_H, high_V, high_S))

        if show == True:
            cv.imshow(self.var_HSV_range_1.window_detection_name, frame_threshold)

        return  frame_threshold, [low_H, low_S, low_V, high_H, high_S, high_V]


    def HSV_adjustment(self, img, factor_H, factor_S, factor_V):

        frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # checkpoint to continue
        # purpose of this function is adjust like lighrroom

    def sharpen(self,img, show = True):
        '''
        sharpen : using factor to adjust
        :param img:
        :param show:
        :return: img, (factor)
        '''
        factor = self.var_sharpen.return_var()
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        kernel = (factor/10) * kernel
        img = cv.filter2D(img, -1, kernel)
        if show == True:
            cv.imshow(self.var_sharpen.window_sharp_name, img)
        # checkpoint to continue
        return img, (factor)


    def line_detection(self, img, draw_img, show = True):
        '''
        Line detection
        # with the following arguments:
        # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        # lines: A vector that will store the parameters (r,θ) of the detected lines
        # rho : The resolution of the parameter r in pixels. We use 1 pixel. ( 1 to 10 )
        # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180) ( 30 - 180 )
        # threshold: The minimum number of intersections to "*detect*" a line
        # srn and stn: Default parameters to zero. Check OpenCV reference for more info.
        # Draw the lines
        :param img:
        :param draw_img:
        :param show:
        :return: draw_img,lines , (rho1, theta2, threshold3, none4, srn5, stn6)
        '''

        # copy_img = img.copy()
        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # copy_img = img.copy()
        rho1, theta2, threshold3, none4, srn5, stn6 = self.var_line_det.return_var()
        if rho1 == 0:
            rho1 = 1
        if theta2 == 0:
            theta2 = 1
        lines = cv.HoughLines(img, rho1, np.pi / theta2, threshold3, None, srn5, stn6)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(draw_img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        if show == True:
            cv.imshow(self.var_line_det.window_line_detection_name, draw_img)

        return draw_img,lines , (rho1, theta2, threshold3, none4, srn5, stn6)

    def circle_detection(self, img,draw_img, show= True):
        '''
        Circle detection : to detect circle, this can adjust 4 params
        param 1 , param 2 , min, max
        :param img:
        :param draw_img:
        :param show:
        :return: image, list of circle, all of params
        '''

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rows =  img.shape[0]

        param1,param2, min,max = self.var_circle_det.return_var()

        if param1 == 0:
            param1 = 1
        if param2 == 0:
            param2 = 1
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, rows / 5,
                                   param1=param1, param2=param2,
                                   minRadius=min, maxRadius=max)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(draw_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(draw_img, center, radius, (255, 0, 255), 3)


        if show == True:
            cv.imshow(self.var_circle_det.window_circle_det_name, draw_img)

        return img, circles, (param1,param2, min, max)

    def circle_detection_1(self, img,draw_img, show= True):
        '''
        Circle detection : to detect circle, this can adjust 4 params
        param 1 , param 2 , min, max
        :param img:
        :param draw_img:
        :param show:
        :return: image, list of circle, all of params
        '''

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rows =  img.shape[0]

        param1,param2, min, max = self.var_circle_det_1.return_var()

        if param1 == 0:
            param1 = 1
        if param2 == 0:
            param2 = 1
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, rows / 5,
                                   param1=param1, param2=param2,
                                   minRadius=min, maxRadius=max)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(draw_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(draw_img, center, radius, (255, 0, 255), 3)


        if show == True:
            cv.imshow(self.var_circle_det_1.window_circle_det_name, draw_img)

        return img, circles, (param1,param2, min, max)

    def dilate(self, img, show = True):
        '''
        dialation function : making white parts bigger follow kernel shape and size

        type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE
        :param img:
        :param show:
        :return: image  and  kernel_size, type_kernel
        '''
        print("Note : \n type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE")
        kernel_size, type_kernel = self.var_dilate.return_var()

        if type_kernel == 1:
            type_kernel = cv.MORPH_RECT #ok
        elif type_kernel == 2:
            type_kernel = cv.MORPH_OPEN #ok
        elif type_kernel == 3:
            type_kernel = cv.MORPH_CROSS #ok
        elif type_kernel == 4:
            type_kernel = cv.MORPH_DILATE #ok
        elif type_kernel == 5:
            type_kernel = cv.MORPH_ERODE #ok
        elif type_kernel == 6:
            type_kernel = cv.MORPH_ELLIPSE #ok
        else:
            type_kernel = cv.MORPH_ELLIPSE

        if kernel_size == 0:
            kernel_size = 1
        kernel = cv.getStructuringElement(type_kernel, (kernel_size, kernel_size))

        dialate = cv.dilate(img, kernel, iterations=1)

        if show == True:
            cv.imshow(self.var_dilate.window_dilate_det_name, dialate)

        return dialate, (kernel_size, type_kernel)


    def erode(self, img, show = True):
        '''
        erosion function : making black parts bigger follow kernel shape and size

        type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE
        :param img:
        :param show:
        :return: kernel_size, type_kernel
        '''
        print("Note : \n type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE")
        kernel_size, type_kernel = self.var_erode.return_var()
        # "ty:1REC,2GRA,3Cro,4DIA,5SQR,6STA,7ELIP"
        if type_kernel == 1:
            type_kernel = cv.MORPH_RECT  # ok
        elif type_kernel == 2:
            type_kernel = cv.MORPH_OPEN  # ok
        elif type_kernel == 3:
            type_kernel = cv.MORPH_CROSS  # ok
        elif type_kernel == 4:
            type_kernel = cv.MORPH_DILATE  # ok
        elif type_kernel == 5:
            type_kernel = cv.MORPH_ERODE  # ok
        elif type_kernel == 6:
            type_kernel = cv.MORPH_ELLIPSE  # ok
        else:
            type_kernel = cv.MORPH_ELLIPSE

        if kernel_size == 0 :
            kernel_size = 1
        kernel = cv.getStructuringElement(type_kernel, (kernel_size, kernel_size))
        # kernel = cv.getStructuringElement(type_kernel, (2 * kernel_size + 1, 2 * kernel_size + 1),
        #                                    (kernel_size, kernel_size))
        erode = cv.erode(img, kernel)
        if show == True:
            cv.imshow(self.var_erode.window_erode_det_name, erode)

        return erode, (kernel_size, type_kernel)

    def sobel(self,img, show =True):
        '''
        sobel is algorithm using derivitive of image
        :param img:
        :param show:
        :return:
        '''
        kernel_size, delta_val, scale_val = self.var_sobel.return_var()
        ddepth = cv.CV_16S
        if not(kernel_size > 0 and kernel_size%2== 1):
            kernel_size +=1
        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=kernel_size, scale=scale_val, delta=delta_val, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=kernel_size, scale=scale_val, delta=delta_val, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


        if show == True:
            cv.imshow(self.var_sobel.window_sobel_det_name, grad)
        
        return grad, (kernel_size, delta_val, scale_val)