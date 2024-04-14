from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os, glob



def calculate_distance(img, dist=cv.DIST_L2, kernel_size=5):
    """
    calculate the distance transform of the given img

    :return:
    """

    # img = cv.imread(img_path)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    # _, threshold = cv.threshold(grey, 123, 255, cv.THRESH_BINARY)
    # cv.imshow("thres", threshold)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    distTransform = cv.distanceTransform(grey, dist, kernel_size)
    # distTransform.astype('uint8')
    return distTransform

def show_destroy(img):
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#
# mask_path = 'testing/bowl1_annotated/999579001.jpg'
# img = cv.imread(mask_path)
# print(f'img: {img.shape}')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     #357159
#
# print(img.shape)
#
# distTransform= cv.distanceTransform(img, cv.DIST_L2, 5)
# cv.imshow("img", img)
# cv.imshow("distance", distTransform)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
