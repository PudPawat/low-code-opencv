import cv2
import os



source = "F:\Ph.D\mirror\data\mirror_defect2"
save = "F:\Ph.D\mirror\data\mirror_defect2_crop"
im_name = os.listdir(source)
print(im_name)
for name in im_name:
    im = cv2.imread(os.path.join(source,name))
    # Select
    factor = 0.3
    im_crop_factor = cv2.resize(im,(int(im.shape[1]*factor),int(im.shape[0]*factor)))
    r = cv2.selectROI(im_crop_factor)
    # print("wire : "+str(i)+" coord: ",r)
    # Crop image

    imCrop = im[int(r[1]/factor):int((r[1] + r[3])/factor), int(r[0]/factor):int((r[0] + r[2])/factor)]
    cv2.imwrite(os.path.join(save,name),imCrop)