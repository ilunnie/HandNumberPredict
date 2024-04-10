import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

from utils import SaveImage

def fft(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    return img

def ifft(fimg):
    fimg = np.fft.ifftshift(fimg)
    fimg = np.fft.ifft2(fimg)
    return fimg

def mag(img):
    absvalue = np.abs(img)
    magnitude = 20 * np.log(absvalue)
    return magnitude

def norm(img):
    img = cv.normalize(
        img, None, 0, 255,
        cv.NORM_MINMAX
    )
    
def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def Fourier(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = fft(img)
    img = mag(img)
    norm(img)
    
    return img

if __name__ == "__main__":
    path = 'images/normal'
    save_path = 'images/fourier'
    
    for dir in os.listdir(path):
        dir_path = f'{path}/{dir}'
        for file in os.listdir(dir_path):
            img = Fourier(f'{dir_path}/{file}')
            SaveImage(f'{save_path}/{dir}', file, img)