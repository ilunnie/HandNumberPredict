import os
import cv2 as cv

def SaveImage(path, file, img):
    if not os.path.exists(path):
        os.makedirs(path)
    cv.imwrite(f'{path}/{file}', img)