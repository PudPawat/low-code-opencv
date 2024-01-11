import pytesseract  
import os
import cv2

path = "../data/CONTACTLENS_READTXT/"
names = os.listdir(path)
print(names)
for name in names:
    read_im_path = os.path.join(path,name)
    # print(im)
    image = cv2.imread(read_im_path)

    boxes = pytesseract.image_to_boxes(image)
    print(boxes, type(boxes))


    str_read = pytesseract.image_to_string(image)
    cv2.putText(image,str_read,(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    print("--"*10,"\n"*2,str_read)
    cv2.imwrite("../output/CONTACTLENS_READTXT/result"+name,image)
    cv2.imshow("asd", image)
    cv2.waitKey(1)


