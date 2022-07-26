import numpy as np
from PIL import Image

import sys
import argparse
import os.path

from csdlib import csdlib as csd


# Doing the main work here.
if __name__ == "__main__":

    # 1 handling the in line parameters
    parser = argparse.ArgumentParser(
        description="IMG to CSD converter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-x", "--width", type=float, default=500.0)
    parser.add_argument("-y", "--height", type=float, default=500.0)

    parser.add_argument("image", help="Image input file")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    source_img = config["image"]
    w = config["width"]
    h = config["height"]

    if os.path.isfile(source_img):
        # loading the picture file
        try:
            loaded_img = Image.open(source_img)
        except:
            print(f"Problem opening {source_img} file!")
            sys.exit(1)
    else:
        print(f"Can't find the {source_img} file")
        sys.exit(1)

    # converting the image to the array
    array_img = np.array(loaded_img)
    print(array_img.shape)

    pixels_x = array_img.shape[0]
    pixels_y = array_img.shape[1]

    output_square_size = max(pixels_x, pixels_y)

    XSecArray = np.zeros((output_square_size, output_square_size))

    for x in range(pixels_x):
        for y in range(pixels_y):
            result = 0
            # R = array_img[x,y,0]
            # G = array_img[x,y,1]
            # B = array_img[x,y,2]

            rgb = np.array(array_img[x, y])

            if rgb.sum() < 3 * 255:
                result = 1 + np.where(rgb == np.amax(rgb))[0][0]
            XSecArray[x, y] = result

    print(XSecArray)
    print(XSecArray.shape)

    dXmm = dYmm = min(w / pixels_x, h / pixels_y)

    print(f"dX: {dXmm} mm,\t dY: {dYmm} mm")

    output_csd_file = source_img[:-4] + ".csd"

    S = csd.cointainer(XSecArray, dXmm, dYmm)
    S.save(output_csd_file)
