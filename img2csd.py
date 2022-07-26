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
            # R = array_img[x, y, 0]
            # G = array_img[x, y, 1]
            # B = array_img[x, y, 2]

            # if R < 255 or G < 255 or B < 255:
            #     if abs(R - G) < 20 and B > 40:
            #         result = 3
            #     elif abs(R - B) < 20 and G > 40:
            #         result = 2
            #     elif abs(G - B) < 20 and R > 40:
            #         result = 1
            rgb = np.array(array_img[x, y])
            if rgb.sum() < 3 * 255:
                result = 1 + np.where(rgb == np.amax(rgb))[0][0]
            XSecArray[x, y] = result

    print(XSecArray)
    print(XSecArray.shape)

    dXmm = dYmm = min(w / pixels_x, h / pixels_y)

    print(f"dX: {dXmm} mm,\t dY: {dYmm} mm")

    if config["optimize"]:
        print()
        print("Simplifying the geometry...", end="")
        splits = 1
        for _ in range(4):
            print(f"...{splits}", end="")
            if dXmm < 1 or dYmm < 1 or max(XSecArray.shape) > config["maxsize"]:

                XSecArray = XSecArray[::2, ::2]

                dXmm = dXmm * 2
                dYmm = dYmm * 2

            else:
                print()
                print("No further subdivisions make sense")
                break

        print()
        print(f"dX:{dXmm}mm dY:{dYmm}mm")
        print(f"Data table size: {XSecArray.shape}")

    output_csd_file = source_img[:-4] + ".csd"
    dXmm = round(dXmm, 2)
    dYmm = round(dYmm, 2)

    S = csd.cointainer(XSecArray, dXmm, dYmm)
    S.save(output_csd_file)
