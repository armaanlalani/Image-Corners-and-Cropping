import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian(x, stddev):
    return 1 / np.sqrt(2 * np.pi) * stddev * np.exp(-(x ** 2) / (2 * stddev ** 2)) # gaussian function

def compute_kernel(sigma):
    dim = 2 * int(2 * sigma + 0.5) + 1 # determines the size of the kernel using sigma --> ensures the size of the kernel is odd
    k_gaussian = np.linspace(-(dim//2), dim//2, dim) # creates the array that will be used to create the outer product
    for i in range(len(k_gaussian)):
        k_gaussian[i] = gaussian(k_gaussian[i], sigma) # 1D gaussian function
    k_gaussian = np.outer(k_gaussian.T, k_gaussian.T) # creates a 2D gaussian function by taking the outer product of the two 1D gaussian functions
    k_gaussian = k_gaussian / k_gaussian.max() # normalizes the kernel to ensure the maximum value is 1
    return k_gaussian

def convolve(im, kern):
    im_h = im.shape[0]
    im_w = im.shape[1]
    kern_h = kern.shape[0]
    kern_w = kern.shape[1]
    output = np.zeros((im_h, im_w))

    kern_size = kern_h * kern_w

    add = [int((kern_h-1)/2), int((kern_w-1)/2)] # size of the additional padded height and weight when filter is placed at edges of the image
    new_im = np.zeros((im_h + 2 * add[0], im_w + 2 * add[1])) # dimensions of the padded image
    new_im[add[0] : im_h + add[0], add[1] : im_w + add[1]] = im # sets the non-padded pixels of the padded image to the pixels of the image being convolved

    for i in range(im_h):
        for j in range(im_w):
            result = kern * new_im[i : i + kern_h, j : j + kern_w] # elementwise multiplication of kernel and appropriate pixels
            output[i, j] = np.sum(result) # adds the elements of the elementwise multiplication
    output = output / kern_size # reduction of pixel values based on kernel size

    return output

if __name__ == "__main__":
    image = Image.open('img1.jpg') # loads the image
    image = image.convert(mode='L') # converts image to grayscale

    data = np.asarray(image) # converts image to a numpy array

    cornersx = []
    cornersy = []

    sigma = 1
    gaussian = compute_kernel(sigma) # computes the gaussian kernel based on the inputted sigma

    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel x
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # sobel y

    grad_x = convolve(data, gx)
    grad_y = convolve(data, gy)

    grad_xx = grad_x**2
    grad_xy = grad_x*grad_y
    grad_yy = grad_y**2

    wx = convolve(grad_xx, gaussian)
    wxy = convolve(grad_xy, gaussian)
    wy = convolve(grad_yy, gaussian)

    eig1 = []
    eig2 = []

    for i in range(grad_xx.shape[0]):
        for j in range(grad_xx.shape[1]):
            M = np.array([[wx[i,j], wxy[i,j]], [wxy[i,j], wy[i,j]]])
            eig = np.linalg.eigvals(M)
            eig1.append(eig[0])
            eig2.append(eig[1])
            if min(eig[0], eig[1]) > 250:
                cornersx.append(i)
                cornersy.append(j)

    plt.scatter(eig1, eig2)
    plt.show()

    fig1, ax1 = plt.subplots(1)
    ax1.scatter(eig1,eig2)

    fig2, ax2 = plt.subplots(1)
    ax2.imshow(image)
    ax2.scatter(cornersy,cornersx,c='#FF0000')

    plt.show()