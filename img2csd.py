import numpy as np

import sys
import argparse
import os.path

from numba import jit, njit


from csdlib import csdlib as csd


# making the NUMBA decorators optional
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


try:
    from numba import njit

    use_njit = True
except ImportError:
    use_njit = False

verbose = not True


def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, args, kwargs)


def getArgs():

    # 1 handling the in line parameters
    parser = argparse.ArgumentParser(
        description="""\
            IMG to CSD converter.
            Converts 2D images like PNG or JPG to the CSD data file.
            Returns the .cds file with the same name and location as input image.

            Info:
            Use on the picture colors to determine phases: 
            Red Color as phase A
            Green as phase B
            Blue as phase C

            JPEG files due to the compression may lead to artifacts and need for
            cleaning in the CSD.py.
            Preferably use .PNG format.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-x", "--width", type=float, default=500.0, help="Canvas Y size in [mm]"
    )
    parser.add_argument(
        "-y", "--height", type=float, default=500.0, help="Canvas Y size in [mm]"
    )

    parser.add_argument(
        "-m",
        "--maxsize",
        type=int,
        default=150,
        help="Output canvas max size in elements if optimization is used",
    )

    parser.add_argument(
        "-opt",
        "--optimize",
        action="store_false",
        help="Disables the optimization of output size.",
    )

    parser.add_argument(
        "-pil",
        "--usepil",
        action="store_true",
        help="Switch to use PIL to load images instead of Matplotlib.",
    )

    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Display the loaded image before proceed.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display the detailed information along process.",
    )

    parser.add_argument("image", help="Image input file")
    args = parser.parse_args()

    return vars(args)


# @njit
@conditional_decorator(njit, use_njit)
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

            if 25 < rgb.sum() < 3 * 255:
                result = 1 + np.where(rgb == np.amax(rgb))[0][0]
            XSecArray[x, y] = result

    return XSecArray


# @njit
@conditional_decorator(njit, use_njit)
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
    # myLog(f"Size after cropping: {new_size}x{new_size}")
    # myLog(f"Cropping: top {crop_top}\t btm {crop_btm}")
    # myLog(f"Cropping: lft {crop_left}\t rgt {crop_right}")

    crop_XSecArray = np.zeros((new_size, new_size))
    crop_XSecArray[
        1 : 1 + size_y - crop_top - crop_btm, 1 : 1 + size_x - crop_left - crop_right
    ] = XSecArray[crop_top:-crop_btm, crop_left:-crop_right]

    return crop_XSecArray


def loadImageFromFile(config):

    source_img = config["image"]
    w = config["width"]
    h = config["height"]

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
@conditional_decorator(njit, use_njit)
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


def main():
    global verbose

    config = getArgs()

    verbose = config["verbose"]
    source_img = config["image"]
    w = config["width"]
    h = config["height"]
    optimize = config["optimize"]

    myLog(config)

    array_img = loadImageFromFile(config)
    myLog(array_img.shape)
    pixels_x = array_img.shape[0]
    pixels_y = array_img.shape[1]

    dXmm = dYmm = min(w / pixels_x, h / pixels_y)

    XSecArray = getCSD(array_img)
    myLog(XSecArray)
    myLog(XSecArray.shape)

    if np.sum(XSecArray) == 0:
        myLog("No cross section data found. No output generated.")
        sys.exit(1)

    XSecArray = trimEmpty(XSecArray)
    myLog("Trimming the empty space...")
    myLog(XSecArray.shape)
    myLog(f"dX: {dXmm} mm,\t dY: {dYmm} mm")

    if optimize:
        myLog()
        myLog("Simplifying the geometry...", end="")
        XSecArray, dXmm, dYmm, splits = simplify(
            XSecArray, dXmm, dYmm, config["maxsize"]
        )

        myLog(f"done {splits}", end="")
        myLog()
        myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
        myLog(f"Data table size: {XSecArray.shape}")

    output_csd_file = source_img[:-4] + ".csd"
    dXmm = round(dXmm, 2)
    dYmm = round(dYmm, 2)

    S = csd.cointainer(XSecArray, dXmm, dYmm)
    S.save(output_csd_file)


if __name__ == "__main__":
    main()
