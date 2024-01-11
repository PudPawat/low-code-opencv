import cv2 as cv


class TrackBar(object):
    def __init__(self):
        pass

    class Binary(object):
        def __init__(self):
            self.window_binary_name = 'binary'
            self.binary_th_max = 500
            self.binary_th1 = 10
            self.binary_th1_name = 'th1'

            # binary
            cv.namedWindow(self.window_binary_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.binary_th1_name, self.window_binary_name, self.binary_th1,
                              self.binary_th_max,
                              self.on_binary_th1)

        # binary
        def on_binary_th1(self, val):
            self.binary_th1 = val
            cv.setTrackbarPos(self.binary_th1_name, self.window_binary_name, self.binary_th1)


        def return_var(self):
            return self.binary_th1


    class Canny(object):

        def __init__(self):
            self.canny_max = 3000
            self.Y_canny = 10
            self.X_canny = 10
            self.window_canny_name = 'Canny'
            self.Y_canny_name = 'Y_Canny'
            self.X_canny_name = 'X_Canny'

            cv.namedWindow(self.window_canny_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.X_canny_name, self.window_canny_name, self.X_canny,
                              self.canny_max,
                              self.on_X_canny)
            cv.createTrackbar(self.Y_canny_name, self.window_canny_name, self.Y_canny,
                              self.canny_max,
                              self.on_Y_canny)

        def on_Y_canny(self, val):
            self.Y_canny = val
            cv.setTrackbarPos(self.Y_canny_name, self.window_canny_name, self.Y_canny)

        def on_X_canny(self, val):
            self.X_canny = val
            cv.setTrackbarPos(self.X_canny_name, self.window_canny_name, self.X_canny)

        def return_var(self):
            return (self.Y_canny, self.X_canny)

    class Canny1(object):

        def __init__(self):
            self.canny_max = 3000
            self.Y_canny = 10
            self.X_canny = 10
            self.window_canny_name = 'Canny_1'
            self.Y_canny_name = 'Y_Canny'
            self.X_canny_name = 'X_Canny'

            cv.namedWindow(self.window_canny_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.X_canny_name, self.window_canny_name, self.X_canny,
                              self.canny_max,
                              self.on_X_canny)
            cv.createTrackbar(self.Y_canny_name, self.window_canny_name, self.Y_canny,
                              self.canny_max,
                              self.on_Y_canny)

        def on_Y_canny(self, val):
            self.Y_canny = val
            cv.setTrackbarPos(self.Y_canny_name, self.window_canny_name, self.Y_canny)

        def on_X_canny(self, val):
            self.X_canny = val
            cv.setTrackbarPos(self.X_canny_name, self.window_canny_name, self.X_canny)

        def return_var(self):
            return (self.Y_canny, self.X_canny)

    class Blur(object):

        def __init__(self):
            self.window_blur_name = 'Blur'
            self.blur_max = 100
            self.blur1 = 1
            self.blur_1_name = 'blur_1'

            cv.namedWindow(self.window_blur_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.blur_1_name, self.window_blur_name, self.blur1,
                              self.blur_max,
                              self.on_blur_1)

        # blur
        def on_blur_1(self, val):
            self.blur1 = val
            cv.setTrackbarPos(self.blur_1_name, self.window_blur_name, self.blur1)

        def return_var(self):
            return self.blur1

    class GaussianBlur(object):

        def __init__(self):
            self.window_blur_name = 'GaussianBlur'
            self.blur_max = 100
            self.blur_y = 1
            self.blur_x = 1
            self.blur_x_name = 'blur x'
            self.blur_y_name = "blur y"

            cv.namedWindow(self.window_blur_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.blur_x_name, self.window_blur_name, self.blur_x,
                              self.blur_max,
                              self.on_blur_x)
            cv.createTrackbar(self.blur_y_name, self.window_blur_name, self.blur_y,
                              self.blur_max,
                              self.on_blur_y)

        # blur
        def on_blur_x(self, val):
            self.blur_x = val
            cv.setTrackbarPos(self.blur_x_name, self.window_blur_name, self.blur_x)

        def on_blur_y(self, val):
            self.blur_y = val
            cv.setTrackbarPos(self.blur_y_name, self.window_blur_name, self.blur_y)

        def return_var(self):
            return (self.blur_x, self.blur_y)

    class Sharpen(object):

        def __init__(self):
            self.window_sharp_name = 'sharpen'
            self.sharp_max = 20
            self.sharp = 1
            self.sharp_1_name = 'sharp_val'

            cv.namedWindow(self.window_sharp_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.sharp_1_name, self.window_sharp_name, self.sharp,
                              self.sharp_max,
                              self.on_sharp_1)

        def on_sharp_1(self, val):
            self.sharp = val
            cv.setTrackbarPos(self.sharp_1_name, self.window_sharp_name, self.sharp)

        def return_var(self):
            return self.sharp

    class HSV(object):
        def __init__(self):
            self.max_value = 255
            self.max_value_H = 360 // 2
            self.low_H = 0
            self.low_S = 0
            self.low_V = 56
            self.high_H = self.max_value_H
            self.high_S = self.max_value
            self.high_V = self.max_value

            self.window_capture_name = 'Video Capture'
            self.window_detection_name = 'Object Detection'

            self.low_H_name = 'Low H'
            self.low_S_name = 'Low S'
            self.low_V_name = 'Low V'
            self.high_H_name = 'High H'
            self.high_S_name = 'High S'
            self.high_V_name = 'High V'

            cv.namedWindow(self.window_detection_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.low_H_name, self.window_detection_name, self.low_H, self.max_value_H,
                              self.on_low_H_thresh_trackbar)
            cv.createTrackbar(self.high_H_name, self.window_detection_name, self.high_H, self.max_value_H,
                              self.on_high_H_thresh_trackbar)
            cv.createTrackbar(self.low_S_name, self.window_detection_name, self.low_S, self.max_value,
                              self.on_low_S_thresh_trackbar)
            cv.createTrackbar(self.high_S_name, self.window_detection_name, self.high_S, self.max_value,
                              self.on_high_S_thresh_trackbar)
            cv.createTrackbar(self.low_V_name, self.window_detection_name, self.low_V, self.max_value,
                              self.on_low_V_thresh_trackbar)
            cv.createTrackbar(self.high_V_name, self.window_detection_name, self.high_V, self.max_value,
                              self.on_high_V_thresh_trackbar)

        def on_low_H_thresh_trackbar(self, val):
            self.low_H = val
            self.low_H = min(self.high_H - 1, self.low_H)
            cv.setTrackbarPos(self.low_H_name, self.window_detection_name, self.low_H)

        def on_high_H_thresh_trackbar(self, val):
            self.high_H = val
            self.high_H = max(self.high_H, self.low_H + 1)
            cv.setTrackbarPos(self.high_H_name, self.window_detection_name, self.high_H)

        def on_low_S_thresh_trackbar(self, val):
            self.low_S = val
            self.low_S = min(self.high_S - 1, self.low_S)
            cv.setTrackbarPos(self.low_S_name, self.window_detection_name, self.low_S)

        def on_high_S_thresh_trackbar(self, val):
            self.high_S = val
            self.high_S = max(self.high_S, self.low_S + 1)
            cv.setTrackbarPos(self.high_S_name, self.window_detection_name, self.high_S)

        def on_low_V_thresh_trackbar(self, val):
            self.low_V = val
            self.low_V = min(self.high_V - 1, self.low_V)
            cv.setTrackbarPos(self.low_V_name, self.window_detection_name, self.low_V)

        def on_high_V_thresh_trackbar(self, val):
            self.high_V = val
            self.high_V = max(self.high_V, self.low_V + 1)
            cv.setTrackbarPos(self.high_V_name, self.window_detection_name, self.high_V)

        def return_var(self):
            return (self.low_H, self.low_S, self.low_V, self.high_H, self.high_S, self.high_V)
    # binary
    # def on_binary_th1(self, val):
    #     print(val)
    #     self.binary_th1 = val
    #     cv.setTrackbarPos(self.binary_th1_name, self.window_binary_name, self.binary_th1)
    #
    # def on_binary_th2(self, val):
    #     self.binary_th2 = val
    #     cv.setTrackbarPos(self.binary_th2_name, self.window_binary_name, self.binary_th2)

    class HSV1(object):
        def __init__(self):
            self.max_value = 255
            self.max_value_H = 360 // 2
            self.low_H = 0
            self.low_S = 0
            self.low_V = 0
            self.high_H = self.max_value_H
            self.high_S = self.max_value
            self.high_V = self.max_value

            self.window_capture_name = 'Video Capture_HSV_1'
            self.window_detection_name = 'Object Detection_1'

            self.low_H_name = 'Low H'
            self.low_S_name = 'Low S'
            self.low_V_name = 'Low V'
            self.high_H_name = 'High H'
            self.high_S_name = 'High S'
            self.high_V_name = 'High V'

            cv.namedWindow(self.window_detection_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.low_H_name, self.window_detection_name, self.low_H, self.max_value_H,
                              self.on_low_H_thresh_trackbar)
            cv.createTrackbar(self.high_H_name, self.window_detection_name, self.high_H, self.max_value_H,
                              self.on_high_H_thresh_trackbar)
            cv.createTrackbar(self.low_S_name, self.window_detection_name, self.low_S, self.max_value,
                              self.on_low_S_thresh_trackbar)
            cv.createTrackbar(self.high_S_name, self.window_detection_name, self.high_S, self.max_value,
                              self.on_high_S_thresh_trackbar)
            cv.createTrackbar(self.low_V_name, self.window_detection_name, self.low_V, self.max_value,
                              self.on_low_V_thresh_trackbar)
            cv.createTrackbar(self.high_V_name, self.window_detection_name, self.high_V, self.max_value,
                              self.on_high_V_thresh_trackbar)

        def on_low_H_thresh_trackbar(self, val):
            self.low_H = val
            self.low_H = min(self.high_H - 1, self.low_H)
            cv.setTrackbarPos(self.low_H_name, self.window_detection_name, self.low_H)

        def on_high_H_thresh_trackbar(self, val):
            self.high_H = val
            self.high_H = max(self.high_H, self.low_H + 1)
            cv.setTrackbarPos(self.high_H_name, self.window_detection_name, self.high_H)

        def on_low_S_thresh_trackbar(self, val):
            self.low_S = val
            self.low_S = min(self.high_S - 1, self.low_S)
            cv.setTrackbarPos(self.low_S_name, self.window_detection_name, self.low_S)

        def on_high_S_thresh_trackbar(self, val):
            self.high_S = val
            self.high_S = max(self.high_S, self.low_S + 1)
            cv.setTrackbarPos(self.high_S_name, self.window_detection_name, self.high_S)

        def on_low_V_thresh_trackbar(self, val):
            self.low_V = val
            self.low_V = min(self.high_V - 1, self.low_V)
            cv.setTrackbarPos(self.low_V_name, self.window_detection_name, self.low_V)

        def on_high_V_thresh_trackbar(self, val):
            self.high_V = val
            self.high_V = max(self.high_V, self.low_V + 1)
            cv.setTrackbarPos(self.high_V_name, self.window_detection_name, self.high_V)

        def return_var(self):
            return (self.low_H, self.low_S, self.low_V, self.high_H, self.high_S, self.high_V)

    class LineDetection(object):
        def __init__(self):
            self.window_line_detection_name = 'Line detection'
            self.line1 = 1 # 1 to 10 rho
            self.line2 = 180  # np.pi/180 theta
            self.line3 = 1 # 0 to 500 th
            self.line4 = None
            self.line5 = 0 # 0 to 100 srn
            self.line6 = 0 # 0 to 200 stn

            self.max_line1 = 100
            self.max_line3 = 1000
            self.max_value = 200

            self.line1_name = "rho"
            self.line2_name = "theta"
            self.line3_name = "threshold"
            self.line4_name = "None"
            self.line5_name = "srn(def=0)"
            self.line6_name = "stn(def=0)"

            cv.namedWindow(self.window_line_detection_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.line1_name, self.window_line_detection_name, self.line1, self.max_line1,
                              self.on_line_1)
            cv.createTrackbar(self.line2_name, self.window_line_detection_name, self.line2, self.max_value,
                              self.on_line_2)
            # cv.createTrackbar(self.line3_name, self.window_line_detection_name, self.line3, self.max_line3,
            #                   self.on_line_3)
            # cv.createTrackbar(self.line5_name, self.window_line_detection_name, self.line5, self.max_value,
            #                   self.on_line_5)
            # cv.createTrackbar(self.line6_name, self.window_line_detection_name, self.line6, self.max_value,
            #                   self.on_line_6)



        def on_line_1(self, val):
            self.line1 = val
            cv.setTrackbarPos(self.line1_name, self.window_line_detection_name, self.line1)

        def on_line_2(self, val):
            self.line2 = val
            cv.setTrackbarPos(self.line2_name, self.window_line_detection_name, self.line2)

        def on_line_3(self, val):
            self.line3 = val
            cv.setTrackbarPos(self.line3_name, self.window_line_detection_name, self.line3)

        def on_line_5(self, val):
            self.line5 = val
            cv.setTrackbarPos(self.line5_name, self.window_line_detection_name, self.line5)

        def on_line_6(self, val):
            self.line6 = val
            cv.setTrackbarPos(self.line6_name, self.window_line_detection_name, self.line6)


        def return_var(self):
            return (self.line1, self.line2, self.line3, self.line4, self.line5, self.line6)

    class CircleDetection(object):
        def __init__(self):
            self.window_circle_det_name = "Circle"
            self.circle_param1 = 1
            self.circle_param2 = 5
            self.min = 0
            self.max = 20

            self.max_all = 500

            self.circle_param1_name = "param1"
            self.circle_param2_name = "param2"
            self.min_name = "min"
            self.max_name = "max"

            cv.namedWindow(self.window_circle_det_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.circle_param1_name, self.window_circle_det_name, self.circle_param1, self.max_all,
                              self.on_cir_param1)
            cv.createTrackbar(self.circle_param2_name, self.window_circle_det_name, self.circle_param2, self.max_all,
                              self.on_cir_param2)
            cv.createTrackbar(self.min_name, self.window_circle_det_name, self.min, self.max_all,
                              self.on_min_param)
            cv.createTrackbar(self.max_name, self.window_circle_det_name, self.max, self.max_all,
                              self.on_max_param)

        def on_cir_param1(self,val):
            self.circle_param1 = val
            cv.setTrackbarPos(self.circle_param1_name, self.window_circle_det_name, self.circle_param1)

        def on_cir_param2(self,val):
            self.circle_param2 = val
            cv.setTrackbarPos(self.circle_param2_name, self.window_circle_det_name, self.circle_param2)

        def on_min_param(self,val):
            self.min = val
            cv.setTrackbarPos(self.min_name, self.window_circle_det_name, self.min)

        def on_max_param(self,val):
            self.max = val
            cv.setTrackbarPos(self.max_name, self.window_circle_det_name, self.max)

        def return_var(self):
            return (self.circle_param1, self.circle_param2, self.min, self.max)

    class CircleDetection1(object):
        def __init__(self):
            self.window_circle_det_name = "Circle_1"
            self.circle_param1 = 1
            self.circle_param2 = 4
            self.min = 13
            self.max = 65

            self.max_all = 500

            self.circle_param1_name = "param1"
            self.circle_param2_name = "param2"
            self.min_name = "min"
            self.max_name = "max"

            cv.namedWindow(self.window_circle_det_name, cv.WINDOW_NORMALvvvvv)
            cv.createTrackbar(self.circle_param1_name, self.window_circle_det_name, self.circle_param1, self.max_all,
                              self.on_cir_param1)
            cv.createTrackbar(self.circle_param2_name, self.window_circle_det_name, self.circle_param2, self.max_all,
                              self.on_cir_param2)
            cv.createTrackbar(self.min_name, self.window_circle_det_name, self.min, self.max_all,
                              self.on_min_param)
            cv.createTrackbar(self.max_name, self.window_circle_det_name, self.max, self.max_all,
                              self.on_max_param)

        def on_cir_param1(self,val):
            self.circle_param1 = val
            cv.setTrackbarPos(self.circle_param1_name, self.window_circle_det_name, self.circle_param1)

        def on_cir_param2(self,val):
            self.circle_param2 = val
            cv.setTrackbarPos(self.circle_param2_name, self.window_circle_det_name, self.circle_param2)

        def on_min_param(self,val):
            self.min = val
            cv.setTrackbarPos(self.min_name, self.window_circle_det_name, self.min)

        def on_max_param(self,val):
            self.max = val
            cv.setTrackbarPos(self.max_name, self.window_circle_det_name, self.max)

        def return_var(self):
            return (self.circle_param1, self.circle_param2, self.min, self.max)

    class Dilate(object):
        def __init__(self):
            self.window_dilate_det_name = "dilate"
            self.kernel_size = 5
            self.type_kernel = 1
            self.max_size = 100
            self.max_type = 7
            self.kernel_name = "kenel_size"
            self.type_name = "ty:1REC,2GRA,3Cro,4DIA,5SQR,6STA,7ELIP"

            cv.namedWindow(self.window_dilate_det_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.kernel_name, self.window_dilate_det_name, self.kernel_size, self.max_size,
                              self.on_kernel_size)
            cv.createTrackbar(self.type_name, self.window_dilate_det_name, self.type_kernel, self.max_type,
                              self.on_type_kenel)

        def on_kernel_size(self, val):
            self.kernel_size = val
            cv.setTrackbarPos(self.kernel_name, self.window_dilate_det_name, self.kernel_size)

        def on_type_kenel(self, val):
            self.type_kernel = val
            cv.setTrackbarPos(self.type_name, self.window_dilate_det_name, self.type_kernel)

        def return_var(self):
            return (self.kernel_size, self.type_kernel)

    class Erode(object):
        def __init__(self):
            self.window_erode_det_name = "erode"
            self.kernel_size = 5
            self.type_kernel = 5
            self.max_size = 100
            self.max_type = 7
            self.kernel_name = "kenel_size"
            self.type_name = "ty:1REC,2GRA,3Cro,4DIA,5SQR,6STA,7ELIP"

            cv.namedWindow(self.window_erode_det_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.kernel_name, self.window_erode_det_name, self.kernel_size, self.max_size,
                              self.on_kernel_size)
            cv.createTrackbar(self.type_name, self.window_erode_det_name, self.type_kernel, self.max_type,
                              self.on_type_kenel)

        def on_kernel_size(self, val):
            self.kernel_size = val
            cv.setTrackbarPos(self.kernel_name, self.window_erode_det_name, self.kernel_size)

        def on_type_kenel(self, val):
            self.type_kernel = val
            cv.setTrackbarPos(self.type_name, self.window_erode_det_name, self.type_kernel)

        def return_var(self):
            return (self.kernel_size, self.type_kernel)

    class Sobel(object):
        def __init__(self):
            self.window_sobel_det_name = "sobel"
            self.kernel_size = 5
            self.delta_val = 1
            self.scale_val = 1
            # self.type_kernel = 5
            self.max_size = 100
            self.max_kernel_size = 31
            self.max_ddepth = 20
            self.kernel_name = "kenel_size"
            self.delta_name = "delta_size"
            self.scale_name = "scale"

            cv.namedWindow(self.window_sobel_det_name, cv.WINDOW_NORMAL)
            cv.createTrackbar(self.kernel_name, self.window_sobel_det_name, self.kernel_size, self.max_kernel_size,
                              self.on_kernel_size)
            cv.createTrackbar(self.delta_name, self.window_sobel_det_name, self.delta_val, self.max_size,
                              self.on_delta)
            cv.createTrackbar(self.scale_name, self.window_sobel_det_name, self.scale_val, self.max_ddepth,
                              self.on_scale)

        def on_kernel_size(self, val):
            self.kernel_size = val
            cv.setTrackbarPos(self.kernel_name, self.window_sobel_det_name, self.kernel_size)

        def on_delta(self,val):
            self.delta_val = val
            cv.setTrackbarPos(self.delta_name, self.window_sobel_det_name, self.delta_val)

        def on_scale(self,val):
            self.scale_val = val
            cv.setTrackbarPos(self.scale_name, self.window_sobel_det_name, self.scale_val)

        def return_var(self):
            return (self.kernel_size, self.delta_val, self.scale_val)