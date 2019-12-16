import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#########################################################
# functions to plot digits
#########################################################

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.gray,
               interpolation="nearest")
    plt.colorbar()
    # plt.axis("off")


def plot_digits(data, n_samples_row=10):
    images = [image.reshape(28,28) for image in data]
    n_rows = (len(images) - 1) // n_samples_row + 1
    # append empty images if the last row is not complete
    empty_images = n_rows * n_samples_row - len(data)
    images.append(np.zeros((28, 28 * empty_images)))
    # draw row by row
    images_row = []
    for current_row in range(n_rows):
        tmp_row_images = images[current_row * n_samples_row : (current_row + 1) * n_samples_row]
        images_row.append(np.concatenate(tmp_row_images, axis=1))
    # draw all in one image
    image = np.concatenate(images_row, axis=0)
    plt.figure(figsize=(n_samples_row,n_rows))
    plt.imshow(image, cmap = matplotlib.cm.gray)
    plt.colorbar()
    # plt.axis("off")

#########################################################
#########################################################