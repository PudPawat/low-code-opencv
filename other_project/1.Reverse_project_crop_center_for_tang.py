import cv2
import os
import numpy as np
from lib_save import  Imageprocessing, read_save
degree = "240degree"
source = "data/"+degree
source_result = "data/OUTPUT_PUWATs_DATA/"+degree+"/maskrcnn_angle"
output_path = "output/"+degree+"/"
TXT_path = "data/Puwat_New_Result_JAN_10th_fixcoord/"+degree+"/TXT"
params_arrow = {"HSV": [54, 52, 199, 65, 255, 255], "erode": [1, 0], "dilate": [22, 0]}

def recorrect_txt():
    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for name in im_name:
        frame = cv2.imread(source + "/" + name)
        # factor = 0.6
        # frame_shape = np.array(frame.shape[:2]) * factor
        # frame_shape = frame_shape.astype('int32')
        # frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
        # left,top = coordinate_of_cropped_FOV(frame,params)
        left, top = 1036, 819 #1036 819
        # image size should be  y = 921, x = 1167
        factor_x = 1167/640
        factor_y = 921/480

        ### read txt
        correct_lines = []
        print(name[:-4])
        with open(TXT_path +"/"+ name[:-4] +'.txt') as f:
            lines = f.readlines()
            correct_line = []
            for line in lines:
                print(line)
                line = line.split("\n")# avoid \n
                print("line",line)
                ele = line[0].split(" ")
                print("ele",ele)
                for el in ele:
                    print(el)
                cv2.line(frame,(left+int(int(ele[0])*factor_x),top+int(int(ele[1])*factor_y)), (left+int(int(ele[2])*factor_x),top+int(int(ele[3])*factor_y)),(200,0,0),20)
                correct_line = [left+int(int(ele[0])*factor_x),top+int(int(ele[1])*factor_y), left+int(int(ele[2])*factor_x),top+int(int(ele[3])*factor_y)]
                correct_lines.append(correct_line)
                cv2.line(frame,(int(ele[0]),int(ele[1])),(int(ele[2]),int(ele[3])),(200,0,0),thickness=20)
        cv2.imshow("original", frame)
        cv2.waitKey(0)


        with open(output_path +"/"+ name[:-4] +'.txt','w') as t:
            for line in correct_lines:
                for ele in line:
                    t.write(str(ele))
                    t.write(" ")
                t.write("\n")
        # cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        cv2.imshow("original", frame)
        cv2.waitKey(1)

def recoorect_mask2fullImg():
    output_path = "output/"
    im_name = os.listdir(source)
    # resize_img_baseon_FOV(im_name,source,params,output_path)
    for name in im_name:
        print("data/Puwat_New_Result_JAN_10th_fixcoord/"+degree+"/Mask_processed/"+name[0:-4]+"/mask_0.png")
        frame = cv2.imread("data/Puwat_New_Result_JAN_10th_fixcoord/"+degree+"/Mask_processed/"+name[0:-4]+"/mask_0.png")
        cv2.imshow("frame",frame)
        cv2.waitKey(0)
        ori_frame = cv2.imread(source + "/" + name)
        left, top = 1036, 819  # 1036 819
        # image size should be  y = 921, x = 1167
        factor_x = 1167 / 640
        factor_y = 921 / 480
        mask_img = np.zeros((2048, 2592), np.uint8)
        cv2.imshow("black_image", mask_img)
        # cv2.waitKey(0)
        frame_resize = cv2.resize(frame,(1167,921))
        cv2.imshow("test", frame_resize)
        # cv2.waitKey(0)
        if len(frame_resize.shape) == 3:
            frame_resize = cv2.cvtColor(frame_resize,cv2.COLOR_BGR2GRAY)

        mask_img[top:top +921,left:left+1167] = frame_resize
        kernel = np.ones((8, 8), np.uint8)
        mask_img = cv2.erode(mask_img,kernel)
        save_path = output_path+"mask_fullcoord/"+degree+"/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imwrite(save_path+name,mask_img)
        RGB_mask = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        only_orchid = cv2.bitwise_and(ori_frame,RGB_mask)
        cv2.imshow("test",only_orchid)
        cv2.waitKey(1)

        arrows_mask = np.zeros((2048, 2592), np.uint8)
        with open(TXT_path +"/"+ name[:-4] +'.txt') as f:
            lines = f.readlines()

            correct_line = []
            for num_line,line in enumerate(lines):
                each_arrow_mask = np.zeros((2048, 2592), np.uint8)
                print(line)
                line = line.split("\n")# avoid \n
                print("line",line)
                ele = line[0].split(" ")
                print("ele",ele)
                for el in ele:
                    print(el)
                cv2.line(arrows_mask,(left+int(int(ele[0])*factor_x),top+int(int(ele[1])*factor_y)), (left+int(int(ele[2])*factor_x),top+int(int(ele[3])*factor_y)),(255),20)
                cv2.line(each_arrow_mask,(left+int(int(ele[0])*factor_x),top+int(int(ele[1])*factor_y)), (left+int(int(ele[2])*factor_x),top+int(int(ele[3])*factor_y)),(255),20)
                cv2.imwrite(save_path + name[0:-4]+"arrow_"+str(num_line)+".jpg", each_arrow_mask)
                # cv2.imwrite()
                arrow = [(left+int(int(ele[0])*factor_x),top+int(int(ele[1])*factor_y)),(left+int(int(ele[2])*factor_x),top+int(int(ele[3])*factor_y))]
                print("arrow",arrow)
                correct_line.append(arrow)
            print(correct_line)
        orimask_and_arrows = cv2.bitwise_and(arrows_mask,mask_img)
        # cv2.imshow("original", orimask_and_arrows)
        # cv2.waitKey(0)
class ProcForOrchid():
    def __init__(self, ImgL):
        self.img_L = ImgL


    def crop_for_DL(self):
        frame = self.img_L
        # factor = 0.6
        # frame_shape = np.array(frame.shape[:2])  # y,x
        y, x, _ = (frame.shape)
        top = int(y * 0.4)
        left = int(x * 0.4)
        self.top = top
        self.left = left
        print(left, top)  # 1036 819
        bot = int(y * 0.85)
        right = int(x * 0.85)
        crop_img = frame[top:bot, left:right]
        print(crop_img.shape)  ##  y = 921, x = 1167
        self.crop_img_shape = crop_img.shape
        crop_img = cv2.resize(crop_img, (640, 480))
        return  crop_img

    def convert_arrow2orinal(self, path_name):
        '''

        :param path_name: arrow.txt
        :return: [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]]
        '''
        left, top = 1036, 819  # 1036 819
        # image size should be  y = 921, x = 1167
        factor_x = self.crop_img_shape[1] / 640
        factor_y = self.crop_img_shape[0] / 480
        with open(path_name) as f:
            lines = f.readlines()
        correct_line = []
        for num_line, line in enumerate(lines):
            # each_arrow_mask = np.zeros((2048, 2592), np.uint8)
            print(line)
            line = line.split("\n")  # avoid \n
            print("line", line)
            ele = line[0].split(" ")
            print("ele", ele)
            # for el in ele:
            #     print(el)
            arrow = [self.left + int(int(ele[0]) * factor_x), self.top + int(int(ele[1]) * factor_y),
                     self.left + int(int(ele[2]) * factor_x), self.top + int(int(ele[3]) * factor_y)]
            print("arrow", arrow)
            correct_line.append(arrow)

if __name__ == '__main__':
    recoorect_mask2fullImg()
    # recorrect_txt()
#     reading = read_save()
#     # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [5, 0], "dilate": [18, 0]}
#     # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [12, 0], "dilate": [14, 0]}
#     # params = {"HSV": [33, 77, 0, 48, 255, 222], "erode": [5, 0], "dilate": [14, 0]}
#     source = "data/240degree"
#     # source = "output/240degree" ## cropped output
#     source_result = "data/OUTPUT_PUWATs_DATA/240degree/maskrcnn_angle"
#     output_path = "output/240degree/"
#     TXT_path = "data/OUTPUT_PUWATs_DATA/240degree/TXT"
#     # params_arrow = {"HSV": [54, 52, 199, 65, 255, 255], "erode": [1, 0], "dilate": [22, 0]}
#
#     im_name = os.listdir(source)
#     # resize_img_baseon_FOV(im_name,source,params,output_path)
#     for name in im_name:
#         print(name)
#         frame = cv2.imread(source + "/" + name)
#         # factor = 0.6
#         frame_shape = np.array(frame.shape[:2]) # y,x
#         y,x,_ = (frame.shape)
#         top = int(y*0.4)
#         left = int(x*0.4)
#         bot = int(y*0.85)
#         right = int(x*0.85)
#         crop_img = frame[top:bot, left:right]
#         print(crop_img.shape) ##  y = 921, x = 1167
#         crop_img = cv2.resize(crop_img,(640,480))
#         cv2.imshow("test",crop_img)
#         cv2.imwrite(output_path+name,crop_img)
#         cv2.waitKey(1)
#
#
#         # width = 640
#         # height = 480
#         # left = int(cX - int(width / 2))
#         # top = int(cY - height / 2)
#         try:
#             ## find arrow from tang's result
#             result_mask = arrow_mask(source_result,name,params_arrow)
#             print("asdasd",result_mask.shape)
#
#             ## make a mask for pasting arrow mask on to the big image
#             img_black = np.zeros(frame_shape)
#             img_black[top:top + height, left:left + width] = result_mask
#
#
#             ## resize back to normal size
#             img_black = cv2.resize(img_black,(frame_shape_ori[1],frame_shape_ori[0]))
#             cv2.imshow("mask", img_black)
#             cv2.imshow("Final", img)  # ori_write
#             cv2.imwrite(output_path + name, img_black)
#             key = cv2.waitKey(1)
#         except:
#             pass
