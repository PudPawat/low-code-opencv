import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from other_project.container.warp_and_reverse_warp import *

# img2 = cv.imread('./data/container_focus/chessboard_top3.jpg',0)          # queryImage
# img2 = cv.imread('./data/container/1_A1.jpg',0) # trainImage
def matching_SIFT(img1,img2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for i, (m,n )in enumerate(matches):
        if m.distance < 0.75*n.distance:
            print(i,kp2[i].pt)
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    print("good",len(good),good)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("SIFT", img3)
    cv2.imwrite("SIFT.jpg",img3)
    # plt.imshow(img3),plt.show()
    num_matched_point = len(good)
    return num_matched_point, img3

def matching_SIFT_2(img1,img2):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # print("matches", len(matches),matches)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    matched_count = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            print(i,kp2[i].pt)
            matchesMask[i]=[1,0]
            matched_count +=1
    # print('matches', matched_count)
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # plt.imshow(img3,),plt.show()

    return matched_count, img3

if __name__ == '__main__':
    img1 = cv.imread('./data/container_focus/warp_chessboard_top3.jpg', 0)  # queryImage
    img2 = cv.imread('./data/container_focus/test/C/chessboard_top14.jpg', 0)  # queryImage
    img1 = cv.imread('./data/container_focus/test/C/chessboard_top13.jpg', 0)  # queryImage

    img1 = warp_reverser_warp(img1)
    img2 = warp_reverser_warp(img2)

    save_dir = "data/container/compare/11"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_dir = "data\container\light1" # data_dir contains folders of each class of the data
    # folder_list = os.listdir(data_dir)
    names = os.listdir(data_dir)
    img_ref = cv2.imread("data\container\light1\warp_chessboard_top11.jpg", 0)

    matched_point = []
    for name in names:
        img_comapare = cv2.imread(os.path.join(data_dir,name))

        n_points,img = matching_SIFT(img_ref,img_comapare)
        cv2.putText(img,"Matched: "+str(n_points),(0,img.shape[0]),cv2.FONT_HERSHEY_COMPLEX,1,(200,0,0),2)
        matched_point.append(n_points)
        cv2.imwrite( os.path.join(save_dir,"compare_"+name), img)
    print(names)
    print(matched_point)


