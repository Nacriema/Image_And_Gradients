#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sep 20 15:39:07 2021

@author: Nacriema

Refs:
Compute the image gradient
https://scikit-image.org/skimage-tutorials/lectures/1_image_filters.html

"""
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
from math import ceil
from PIL import Image
'''
^ y-axis
|
|
|
 - - - - > x-axis
'''

# Helper function


def imshow_all(*images, titles=None, cmap=None, ncols=3):
    images = [img_as_float(img) for img in images]
    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=ceil(len(images)/ncols), ncols=ncols, figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)


def partial_dev_along_x_1(image):
    pass


def partial_dev_along_y_1(image):
    gradient_vertical = np.convolve(np.array([-1, 0, 1]), image[:, 0])
    for i in range(1, image.shape[1]):
        gradient_vertical = np.column_stack((gradient_vertical, np.convolve(np.array([-1, 0, 1]), image[:, i])))
    return gradient_vertical


# USING nd image


def partial_dev_along_x_2(image):
    horizontal_kernel = np.array([
        [-1, 0, 1]
    ])
    gradient_horizontal = ndi.correlate(image.astype(float),
                                        horizontal_kernel)
    return gradient_horizontal


def partial_dev_along_y_2(image):
    vertical_kernel = np.array([
        [-1],
        [0],
        [1],
    ])
    gradient_vertical = ndi.correlate(image.astype(float),
                                      vertical_kernel)
    return gradient_vertical


def create_mag_and_ori(mag, grad_x, grad_y):
    """
    Create RGB image that visualize magnitude and orientation simultaneously
    red(x,y) = |mag|.cos(theta), green(x, y) = |mag|.sin(theta), blue(x,y) = 0
    and
    theta = tan-1(grad_x/grad_y)
    :param grad_x:
    :param grad_y:
    :return: rgb in numpy format
    """
    theta = np.abs(np.arctan(grad_x/grad_y))
    green = mag * np.sin(theta)
    red = mag * np.cos(theta)
    blue = np.zeros_like(mag)
    rgb = np.dstack((red, green, blue))
    return rgb


def create_directional_image_derivatives(theta, grad_x, grad_y):
    """
    Compute the directional derivative with given the theta direction
    In matrix form, at the (0, 0) point for example:
                                   [cos(theta)
    [grad_x(0, 0) grad_y(0, 0)]. *
                                    sin(theta)]
    :param theta:
    :param grad_x:
    :param grad_y:
    :return: np array with shape like grad_x
    """

    return grad_x * np.cos(theta) + grad_y * np.sin(theta)


def save_mag_image(mag_and_ori):
    mag_and_ori = np.nan_to_num(mag_and_ori)
    print(np.unique(mag_and_ori))
    im = Image.fromarray(np.uint8(mag_and_ori*255))
    im.save('mag_and_ori_2.jpg')


if __name__ == '__main__':
    image = rgb2gray(imread('./Image/im_1.jpg'))
    # image = Image.open('./Image/im_1.jpg').convert('L')
    # image = np.asarray(image)
    x_grad = partial_dev_along_x_2(image)
    y_grad = partial_dev_along_y_2(image)
    # Compute the magnitude of gradient
    # g = sqrt(g_x^2 + g_y**2)
    g = np.sqrt(x_grad**2 + y_grad**2)
    mag_and_ori = create_mag_and_ori(g, x_grad, y_grad)
    '''
    We need 2 value more, the 45 degree and -45 degree directional derivative
    '''
    grad_45 = create_directional_image_derivatives(theta=(45/180)*(2*np.pi), grad_x=x_grad, grad_y=y_grad)
    grad_neg_45 = create_directional_image_derivatives(theta=(-45/180)*(2*np.pi), grad_x=x_grad, grad_y=y_grad)
    titles = ['original image', 'horizontal gradient', 'vertical gradient', 'magnitude of gradient',
              'derivative_at_45_deg', 'derivative_at_neg_45_deg']
    imshow_all(image, x_grad, y_grad, g, grad_45, grad_neg_45, titles=titles, cmap='gray')
    plt.show()

