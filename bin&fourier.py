import os

from utils import SaveImage
from binarize import Binarize
from fourier import fft, mag, norm

def BinarizeAndFourier(path):
    img = Binarize(path)
    
    img = fft(img)
    img = mag(img)
    norm(img)
    
    return img

if __name__ == "__main__":
    path = 'images/normal'
    save_path = 'images/bin&fourier'
    
    for dir in os.listdir(path):
        dir_path = f'{path}/{dir}'
        for file in os.listdir(dir_path):
            img = BinarizeAndFourier(f'{dir_path}/{file}')
            SaveImage(f'{save_path}/{dir}', file, img)