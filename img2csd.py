import numpy as np

import sys
import argparse

# from csdlib import csdlib as csd
from csdlib import csdcli as csdcli
from csdlib import csdfunctions as csdf


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


def main():
    config = getArgs()

    verbose = config["verbose"]
    source_img = config["image"]
    w = config["width"]
    h = config["height"]
    optimize = config["optimize"]

    csdf.verbose = verbose
    csdf.myLog(config)

    array_img = csdcli.loadImageFromFile(config)

    csdf.myLog(array_img.shape)

    pixels_x = array_img.shape[0]
    pixels_y = array_img.shape[1]

    dXmm = dYmm = min(w / pixels_x, h / pixels_y)

    XSecArray = csdcli.getCSD(array_img)
    csdf.myLog(XSecArray)
    csdf.myLog(XSecArray.shape)

    if np.sum(XSecArray) == 0:
        csdf.myLog("No cross section data found. No output generated.")
        sys.exit(1)

    XSecArray = csdcli.trimEmpty(XSecArray)
    csdf.myLog("Trimming the empty space...")
    csdf.myLog(XSecArray.shape)
    csdf.myLog(f"dX: {dXmm} mm,\t dY: {dYmm} mm")

    if optimize:
        csdf.myLog()
        csdf.myLog("Simplifying the geometry...", end="")
        XSecArray, dXmm, dYmm, splits = csdcli.simplify(
            XSecArray, dXmm, dYmm, config["maxsize"]
        )

        csdf.myLog(f"done {splits}", end="")
        csdf.myLog()
        csdf.myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
        csdf.myLog(f"Data table size: {XSecArray.shape}")

    output_csd_file = source_img[:-4] + ".csd"
    dXmm = round(dXmm, 2)
    dYmm = round(dYmm, 2)

    S = csdf.cointainer(XSecArray, dXmm, dYmm)
    S.save(output_csd_file)


if __name__ == "__main__":
    main()
