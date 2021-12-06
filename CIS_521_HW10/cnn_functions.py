############################################################
# CIS 521: Individual Functions for CNN
############################################################

student_name = "Bowen Chen"

############################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


############################################################

# Include your imports here, if any are used.


############################################################
# Individual Functions
############################################################
def pad_zero(image, image_pad_y, image_pad_x):

    image_y = image.shape[0]

    image_x = image.shape[1]

    padded_image = np.zeros((image_y + 2 * image_pad_y, image_x + 2 * image_pad_x))

    padded_image[image_pad_y:image_pad_y + image_y, image_pad_x:image_pad_x + image_x] = image

    return padded_image


def convolve_greyscale(image, kernel):

    kernel_shape = kernel.shape

    kernel_y = kernel_shape[0]

    kernel_x = kernel_shape[1]

    image_pad_x = (kernel_x - 1) // 2

    image_pad_y = (kernel_y - 1) // 2

    image_padding = pad_zero(image, image_pad_y, image_pad_x)

    kernel_flipped = np.fliplr(np.flipud(kernel))

    result = np.zeros([image.shape[0], image.shape[1]])

    for i in range(image.shape[0]):
        
        for j in range(image.shape[1]):
            
            result[i][j] = np.sum(np.multiply(kernel_flipped, image_padding[i:i + kernel_y, j:j + kernel_x]))

    return result


def convolve_rgb(image, kernel):
    convolved_r = convolve_greyscale(image[:, :, 0], kernel)

    convolved_g = convolve_greyscale(image[:, :, 1], kernel)

    convolved_b = convolve_greyscale(image[:, :, 2], kernel)

    convolved_rgb = np.concatenate(
        (convolved_r[:, :, np.newaxis], convolved_g[:, :, np.newaxis], convolved_b[:, :, np.newaxis]), axis=2)

    return convolved_rgb


def max_pooling(image, kernel, stride):
    
    image_y = image.shape[0]

    image_x = image.shape[1]

    kernel_y = kernel[0]

    kernel_x = kernel[1]

    stride_y = stride[0]

    stride_x = stride[1]

    result = []

    for y in range(0, image_y - kernel_y + 1, stride_y):

        max_pooling_x = []

        for x in range(0, image_x - kernel_x + 1, stride_x):
            
            max_num = np.max(image[y: y + kernel_y, x: x + kernel_x])

            max_pooling_x.append(max_num)

        result.append(max_pooling_x)

    return np.array(result)


def average_pooling(image, kernel, stride):
    
    image_y = image.shape[0]

    image_x = image.shape[1]

    kernel_y = kernel[0]

    kernel_x = kernel[1]

    stride_y = stride[0]

    stride_x = stride[1]

    result = []

    for y in range(0, image_y - kernel_y + 1, stride_y):

        avg_pooling_x = []

        for x in range(0, image_x - kernel_x + 1, stride_x):
            
            avg_num = np.average(image[y: y + kernel_y, x: x + kernel_x])

            avg_pooling_x.append(avg_num)

        result.append(avg_pooling_x)

    return np.array(result)


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


# image = np.array(Image.open('4.1.07.tiff'))
# kernel = np.array([
#     [0.11111111, 0.11111111, 0.11111111],
#     [0.11111111, 0.11111111, 0.11111111],
#     [0.11111111, 0.11111111, 0.11111111]])
# output = convolve_rgb(image, kernel)
# plt.imshow(output.astype('uint8'))
# plt.show()
# print(np.round(output[0:3, 0:3, 0:3], 2))

# image = np.array(Image.open('5.1.09.tiff'))
# plt.imshow(image, cmap='gray')
# plt.show()
# kernel_size = (4, 4)
# stride = (1, 1)
# output = max_pooling(image, kernel_size, stride)
# plt.imshow(output, cmap='gray')
# plt.show()
# print(output.shape)

# x = np.array([0.5, 3, 1.5, -4.7, -100])
# print(sigmoid(x))
