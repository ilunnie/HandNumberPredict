import os
import cv2 as cv
import numpy as np

from utils import SaveImage

def Binarize(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    _, img = cv.threshold(
        img, 100, 255, cv.THRESH_BINARY
    )
    
    img = cv.dilate(img, np.ones((5, 5)))
    img = cv.erode(img, np.ones((3, 3)))
    
    return img

if __name__ == "__main__":
    path = 'images/normal'
    save_path = 'images/binarized'

    for dir in os.listdir(path):
        dir_path = f'{path}/{dir}'
        for file in os.listdir(dir_path):
            img = Binarize(f'{dir_path}/{file}')
            SaveImage(f'{save_path}/{dir}', file, img)