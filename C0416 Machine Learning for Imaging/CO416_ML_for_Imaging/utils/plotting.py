# Copyright 2019, Imperial College London
# 
# Tutorial for CO416 - Machine Learning for Imaging
#
# This file: Functions to plot 2D images.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12




def plot_image(image, interpol="nearest"):
    # image: np.array of one of the following shapes:
    #       grayscale image:    (height, width)
    #       grayscale image:    (height, width, 1)
    #       rgb image:          (height, width, 3)
    print("Plotting image of shape: ", image.shape)
    plt.figure() #(figsize=(n_imgs_per_row*0.5, n_rows*0.5)) # size (width, height), in inches.
    if len(image.shape) == 2:
        fig = plt.imshow(image, cmap="gray", interpolation=interpol) # imshow: (w,h) or (w,h,3)
        plt.colorbar(fig)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        fig = plt.imshow(image[:,:,0], cmap="gray", interpolation=interpol) # imshow: (w,h) or (w,h,3)
        plt.colorbar(fig)
    elif len(image.shape) == 3 and image.shape[2] == 3 :
        _ = plt.imshow(image, interpolation=interpol)
    else:
        raise Error("Wrong shape of given image for plotting.")


def plot_grid_of_images(imgs, n_imgs_per_row=10, interpol="nearest"):
    # imgs: numpy array of one of the following shapes:
    #       grayscales images:  (number-of-images, height, width)
    #       grayscales images:  (number-of-images, height, width, 1)
    #       color images:       (number-of-images, height, width, 3)
    n_rows = 1 + imgs.shape[0] // (n_imgs_per_row + 1)
    
    # Append empty images if the last row is not complete
    n_empty_imgs = n_rows * n_imgs_per_row - imgs.shape[0]
    imgs_to_plot = np.concatenate( [imgs, np.zeros((n_empty_imgs, imgs.shape[1], imgs.shape[2]))], axis=0)
    
    # draw row by row
    row_images = [] # each element will be (image-height, image-width X n_imgs_per_row)
    for current_row in range(n_rows):
        tmp_row_images = imgs_to_plot[current_row * n_imgs_per_row : (current_row + 1) * n_imgs_per_row]
        row_images.append( np.concatenate(tmp_row_images, axis=1) )
    # draw all row-images in one image
    collage_of_images = np.concatenate(row_images, axis=0) # array.shape: (height X n_imgs_per_row, width X n_imgs_per_row)
    
    plot_image(collage_of_images, interpol=interpol)


    
def plot_train_progress(loss_l, acc_train_l, acc_test_l, iters_per_point, total_iters=None):

    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False)
    assert len(loss_l) == len(acc_train_l) == len(acc_test_l)
    x_points = range(0, len(loss_l)*iters_per_point, iters_per_point)
    
    axes[0].plot(x_points, loss_l, color="black", label="Training loss", linewidth=5)
    axes[0].set_title("Training loss", fontsize=10, y=1.022)
    axes[0].yaxis.grid(True, zorder=0)
    axes[0].set_xlabel('Iteration', fontsize=10)
    if total_iters is not None:
        axes[0].set_xlim([0,total_iters])
    axes[0].set_ylim([0,None])
    axes[0].legend(loc='upper right')
    
    axes[1].set_title("Accuracy", fontsize=10, y=1.022)
    axes[1].plot(x_points, acc_train_l, color="blue", label="Train", linewidth=5)
    axes[1].plot(x_points, acc_test_l, color="red", label="Test", linewidth=5)
    axes[1].yaxis.grid(True, zorder=0)
    axes[1].set_xlabel('Iteration', fontsize=10)
    if total_iters is not None:
        axes[1].set_xlim([0,total_iters])
    axes[1].set_ylim([0,100])
    axes[1].legend(loc='lower right')
    
    plt.show()
    
    
    
    