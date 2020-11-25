from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

import os
import collections

def gradient_magnitude(img, sobel_x, sobel_y):
    grad_x = convolve(img, sobel_x) # gradient in the x direction
    grad_y = convolve(img, sobel_y) # gradient in the y direction
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) # magnitude of the gradient
    return grad_magnitude.sum(axis=2) # return the gradient of each colour channel

def grad_remove(im, sobel_x, sobel_y):
    image_h = im.shape[0]
    image_w = im.shape[1]
    gradient = gradient_magnitude(im, sobel_x, sobel_y)

    energy = np.zeros((image_h, image_w))
    energy = gradient # stores the energy of each pixel value

    pos = np.zeros(energy.shape, dtype = np.int) # position of minimum entries of energy
    keep = np.ones((image_h, image_w), dtype = np.bool) # pixels that should be kept versus removed

    for i in range(1,image_h):
        for j in range(image_w):
            if j-1 > 0: # edge case testing
                idx = j-1
            else:
                idx = 0
            min_index = np.argmin(energy[i-1, idx:j+2]) # determines the minimum index from the neighbouring pixels above
            pos[i,j] = j+min_index-1 # updates the position of the minimum energy
            energy[i,j] += energy[i-1, pos[i,j]] # updates the running energy total

    min_index = np.argmin(energy[image_h-1, :]) # determines the index in the bottom row with the lowest energy
    for i in range(image_h-1, -1, -1): # works backwards from the bottom row
        keep[i,min_index] = False
        min_index = pos[i, min_index] # updates the new minimum index to the previous row

    keep = np.stack([keep, keep, keep], axis=2)
    return im[keep].reshape((image_h, image_w-1, 3)) # returns the pixels that are supposed to be kept

def resize(im, height, width, sobel_x, sobel_y):
    image_h = im.shape[0]
    image_w = im.shape[1]
    image = np.zeros((image_h, image_w))
    image = im
    for i in range(image_w - width): # removes the respective number of rows
        image = grad_remove(image, sobel_x, sobel_y)
        print(image.shape)
    image = np.transpose(image, axes=(1,0,2)) # tranposes the image so columns can be removed
    for i in range(image_h - height): # removes the respective number of columns
        image = grad_remove(image, sobel_x, sobel_y)
        print(image.shape)
    image = np.transpose(image, axes=(1,0,2)) # transposes the matrix back to its original
    return image

if __name__ == "__main__":
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel x
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # sobel y

    gx = np.stack([gx, gx, gx], axis=2) # 3D sobel in the x direction
    gy = np.stack([gy, gy, gy], axis=2) # 3D sobel in the y direction

    image = np.array(Image.open(os.path.join(os.getcwd(), 'ex3.jpg')))
    resized = resize(image, 870, 1440, gx, gy)

    image2 = Image.fromarray(resized)
    image2.show()