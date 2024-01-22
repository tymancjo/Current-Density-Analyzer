"""
This is a library file for the procedures and functions related to the use of CLI and more. 
"""

import numpy as np

import sys
import argparse
import os.path

from csdlib import csdlib as csd


# # making the NUMBA decorators optional
# def conditional_decorator(dec, condition):
#     def decorator(func):
#         if not condition:
#             # Return the function unchanged, not decorated.
#             return func
#         return dec(func)

#     return decorator


# try:
#     from numba import njit

#     use_njit = True
# except ImportError:
#     use_njit = False


# @conditional_decorator(njit, use_njit)
def getCSD(array_img):
    pixels_x = array_img.shape[0]
    pixels_y = array_img.shape[1]

    output_square_size = max(pixels_x, pixels_y)

    XSecArray = np.zeros((output_square_size, output_square_size), dtype=np.int8)

    for x in range(pixels_x):
        for y in range(pixels_y):
            result = 0

            r = array_img[x, y, 0]
            g = array_img[x, y, 1]
            b = array_img[x, y, 2]

            rgb = np.array((r, g, b))

            av_color = (r + g + b) / 3
            if r > 1.5 * g and r > 1.5 * b and r > av_color:
                result = 1
            elif g > 1.5 * r and g > 1.5 * b and g > av_color:
                result = 2
            elif b > 1.5 * r and b > 1.5 * g and b > av_color:
                result = 3

            # if 25 < rgb.sum() < 3 * 255:

            # result = 1 + np.where(rgb == np.amax(rgb))[0][0]
            XSecArray[x, y] = result

    return XSecArray


# @njit
# @conditional_decorator(njit, use_njit)
def trimEmpty(XSecArray):
    size_y, size_x = XSecArray.shape
    crop_top = crop_btm = 0
    crop_left = crop_right = 0

    for row in range(size_y):
        crop_top = row
        if np.sum(XSecArray[row, :]) > 0:
            break

    for row in range(size_y):
        crop_btm = row
        if np.sum(XSecArray[-row - 1, :]) > 0:
            break

    for col in range(size_x):
        crop_left = col
        if np.sum(XSecArray[:, col]) > 0:
            break

    for col in range(size_x):
        crop_right = col
        if np.sum(XSecArray[:, -col - 1]) > 0:
            break

    new_size = 3 + max(size_x - crop_left - crop_right, size_y - crop_top - crop_btm)
    print(f"Size after cropping: {new_size}x{new_size}")
    print(f"Cropping: top {crop_top}\t btm {crop_btm}")
    print(f"Cropping: lft {crop_left}\t rgt {crop_right}")

    crop_XSecArray = np.zeros((new_size, new_size))
    crop_XSecArray[
        0 : 0 + size_y - crop_top - crop_btm, 0 : 0 + size_x - crop_left - crop_right
    ] = XSecArray[crop_top:-crop_btm, crop_left:-crop_right]

    return crop_XSecArray


def loadImageFromFile(config):
    source_img = config["image"]
    # w = config["width"]
    # h = config["height"]

    if config["usepil"]:
        from PIL import Image

        if os.path.isfile(source_img):
            # loading the picture file
            try:
                loaded_img = Image.open(source_img)
            except:
                myLog(f"Problem opening {source_img} file!")
                sys.exit(1)
        else:
            myLog(f"Can't find the {source_img} file")
            sys.exit(1)

        if config["show"]:
            loaded_img.show()

        # converting the image to the array
        array_img = np.array(loaded_img)

    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        if os.path.isfile(source_img):
            # loading the picture file
            try:
                plt_img = mpimg.imread(source_img)
            except:
                myLog(f"Problem opening {source_img} file!")
                sys.exit(1)
        else:
            myLog(f"Can't find the {source_img} file")
            sys.exit(1)

        if config["show"]:
            plt.imshow(plt_img)
            plt.show(block=False)

        # converting the image to the array
        # array_img = np.array(loaded_img)
        array_img = np.array(plt_img) * 255

    return np.array(array_img)


# @njit
# @conditional_decorator(njit, use_njit)
def simplify(XSecArray, dXmm, dYmm, maxsize):
    splits = 1
    for _ in range(10):
        if dXmm < 1 or dYmm < 1 or max(XSecArray.shape) > maxsize:
            XSecArray = XSecArray[::2, ::2]

            dXmm = dXmm * 2
            dYmm = dYmm * 2

        else:
            break

    return XSecArray, dXmm, dYmm, splits
