"""
This file is intended to be the Command Line Interface
fot the CSD tool aimed to the quick analysis
for power losses in given geometry.
The idea is to be able to use the saved geometry file
and deliver the required input as a command line
parameters.

As an output the csdf.myLoged info of power losses
is generated on the standard output.
"""

# TODO:
# 1. Read the command line parameters - done
# 2. Loading the main geometry array from the file - done
# 3. Setup the solver - done
# 4. Solve - done
# 5. Prepare results - done
# 6. csdf.myLog results - done
# 7. adding inner code working - done
# 8. cleanu and make use of modules - done

# General imports
import numpy as np
import os.path
import sys
import pickle
import argparse

# Importing local library
from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm


def getArgs():
    """
    Handling the cli line parameters. 
    """
    parser = argparse.ArgumentParser(
        description="CSD cli executor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--size", help="Max single cell size in [mm]", type=float, default=5
    )
    parser.add_argument("-f", "--frequency", type=float, default=50.0)
    parser.add_argument("-T", "--Temperature", type=float, default=140.0)
    parser.add_argument("-l", "--length", type=float, default=1000.0)
    (
        parser.add_argument(
            "-sp", "--simple", action="store_true", help="Show only simple output"
        ),
        parser.add_argument(
            "-csv",
            "--csv",
            action="store_true",
            help="Show only simple output as csv f,dP",
        ),
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Display the detailed information along process.",
        ),
        parser.add_argument(
            "-d",
            "--draw",
            action="store_true",
            help="Draw the graphic window to show the geometry and results.",
        ),
        parser.add_argument(
            "-r",
            "--results",
            action="store_true",
            help="Draw the graphic window with results summary.",
        ),
    )

    parser.add_argument("geometry", help="Geometry description file in .csd format")
    parser.add_argument(
        "current",
        help="Current RMS value for the 3 phase symmetrical analysis in ampers [A]",
        type=float,
    )

    args = parser.parse_args()
    return vars(args)


def main():
    """
    This is the place where the main flow of operation is carried.
    """

    config = getArgs()
    verbose = config["verbose"]
    simple = config["simple"]
    csv = config["csv"]

    # for simplicity so the log procrdure can see it globally
    csdf.verbose = verbose

    csdf.myLog()
    csdf.myLog("Starting operations...")
    csdf.myLog()

    if config["draw"] or config["results"]:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

    XSecArray = np.zeros((0, 0))
    dXmm = dYmm = 1

    # 2 loading the geometry data:
    XSecArray, dXmm, dYmm = csdf.loadTheData(config["geometry"])

    csdf.myLog("Initial geometry array parameters:")
    csdf.myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    csdf.myLog(f"Data table size: {XSecArray.shape}")

    while dXmm > config["size"]:
        csdf.myLog("Splitting the geometry cells...", end="")
        XSecArray = csdm.arraySlicer(inputArray=XSecArray, subDivisions=2)
        dXmm = dXmm / 2
        dYmm = dYmm / 2

    csdf.myLog()
    csdf.myLog("Adjusted geometry array parameters:")
    csdf.myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    csdf.myLog(f"Data table size: {XSecArray.shape}")

    if config["draw"]:
        # making the draw of the geometry in initial state.

        colors = ["white", "red", "green", "blue"]
        cmap = ListedColormap(colors)
        norm = plt.Normalize(vmin=0, vmax=4)

        # Adjust the ticks
        ax = plt.gca()
        num_ticks_x = len(XSecArray[0])
        num_ticks_y = len(XSecArray)

        # Set the ticks and corresponding labels
        step = int(num_ticks_x / (num_ticks_x * dXmm / 10))

        x_ticks = np.arange(0, num_ticks_x, step)
        y_ticks = np.arange(0, num_ticks_y, step)

        # Set the ticks based on the array dimensions
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Set the tick labels by multiplying the tick values by the scaling factor
        ax.set_xticklabels((x_ticks * dXmm).astype(int))
        ax.set_yticklabels((y_ticks * dYmm).astype(int))

        plt.imshow(XSecArray, cmap=cmap, norm=norm)
        plt.show()

        question = input("Do you want to run the analysis? [y]/[n]")
        if question.lower() in ["n", "no", "break", "stop"]:
            sys.exit(0)

    # 3 preparing the solution
    Irms = config["current"]
    # Current vector
    I = [Irms, 120, Irms, 0, Irms, 240]
    f = config["frequency"]
    length = config["length"]
    t = config["Temperature"]

    csdf.myLog()
    csdf.myLog("Starting solver for")

    for k, n in zip([0, 2, 4], ["a", "b", "c"]):
        csdf.myLog(f"I{n} = {I[k]}[A] \t {I[k+1]}[deg] \t {f}[Hz]")

    csdf.myLog()
    csdf.myLog("Complex form:")

    # lets workout the  current in phases as is defined
    in_Ia = I[0] * np.cos(I[1] * np.pi / 180) + I[0] * np.sin(I[1] * np.pi / 180) * 1j
    csdf.myLog(f"Ia: {in_Ia}")

    in_Ib = I[2] * np.cos(I[3] * np.pi / 180) + I[2] * np.sin(I[3] * np.pi / 180) * 1j
    csdf.myLog(f"Ib: {in_Ib}")

    in_Ic = I[4] * np.cos(I[5] * np.pi / 180) + I[4] * np.sin(I[5] * np.pi / 180) * 1j
    csdf.myLog(f"Ic: {in_Ic}")

    vPhA = csdm.arrayVectorize(inputArray=XSecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm)
    vPhB = csdm.arrayVectorize(inputArray=XSecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm)
    vPhC = csdm.arrayVectorize(inputArray=XSecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm)

    elementsVector, elementsPhaseA, elementsPhaseB, elementsPhaseC = csdf.combineVectors(
        vPhA, vPhB, vPhC
    )

    if len(elementsVector) > 1200:
        csdf.myLog()
        csdf.myLog(
            "!!! Size of the elements vector may lead to very long calculation. Be aware!"
        )
        csdf.myLog("You can break the process by CTRL+C")
        csdf.myLog("You may conceder reduce the split steps.")
        csdf.myLog("Optimal element size is around 1.5x1.5mm")
        csdf.myLog()

    if len(elementsVector) > 10000:
        csdf.myLog("Extreme size of elements vector - long calculations immanent!")

    admitanceMatrix = csdm.getGmatrix(
        csdm.getImpedanceArray(
            csdm.getDistancesArray(elementsVector),
            freq=f,
            dXmm=dXmm,
            dYmm=dYmm,
            temperature=t,
            lenght=length,
        )
    )

    # Let's put here some voltage vector
    Ua = complex(1, 0)
    Ub = complex(-0.5, np.sqrt(3) / 2)
    Uc = complex(-0.5, -np.sqrt(3) / 2)

    vA = np.ones(elementsPhaseA) * Ua
    vB = np.ones(elementsPhaseB) * Ub
    vC = np.ones(elementsPhaseC) * Uc

    # voltageVector = np.concatenate((vA, vB, vC), axis=0)
    voltageVector, _, _, _ = csdf.combineVectors(vA, vB, vC)

    # Initial solve
    # Main equation solve
    currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringing each phase current to the assumed Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)

    # expected Ia Ib Ic as symmetrical ones
    # ratios of currents will give us new voltages for phases

    if Ia :
        Ua = Ua * (in_Ia / Ia)
    if Ib:
        Ub = Ub * (in_Ib / Ib)
    if Ic:
        Uc = Uc * (in_Ic / Ic)

    csdf.myLog()
    csdf.myLog("Calculated require Source Voltages")
    csdf.myLog(Ua)
    csdf.myLog(Ub)
    csdf.myLog(Uc)

    # Setting up the voltage vector for final solve
    vA = np.ones(elementsPhaseA) * Ua
    vB = np.ones(elementsPhaseB) * Ub
    vC = np.ones(elementsPhaseC) * Uc

    voltageVector = np.concatenate((vA, vB, vC), axis=0)

    # Final solve
    # Main equation solve
    currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringing each phase current to the assumer Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)
    # end of second solve!

    csdf.myLog()
    csdf.myLog("Solution check...")
    csdf.myLog("Raw Current results:")
    csdf.myLog(f"Ia: {Ia}")
    csdf.myLog(f"Ib: {Ib}")
    csdf.myLog(f"Ic: {Ic}")
    csdf.myLog()
    csdf.myLog(f"Sum: {Ia+Ib+Ic}")

    # Now we normalize up to the expecter I - just a polish
    # as we are almost there with the previous second solve for new VOLTAGES
    modIa = np.abs(Ia)
    modIb = np.abs(Ib)
    modIc = np.abs(Ic)

    if modIa:
        currentPhA *= in_Ia / modIa
    if modIb:
        currentPhB *= in_Ib / modIb
    if modIc:
        currentPhC *= in_Ic / modIc

    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)

    csdf.myLog("Fix Current results:")
    csdf.myLog(f"Ia: {Ia}")
    csdf.myLog(f"Ib: {Ib}")
    csdf.myLog(f"Ic: {Ic}")
    csdf.myLog()
    csdf.myLog(f"Sum: {Ia+Ib+Ic}")

    # Data postprocessing
    getMod = np.vectorize(csdm.getComplexModule)

    resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)
    # for debug
    # csdf.myLog(resultsCurrentVector)
    #
    resultsCurrentVector = getMod(resultsCurrentVector)
    resistanceVector = csdm.getResistanceArray(
        elementsVector, dXmm=dXmm, dYmm=dYmm, temperature=t, lenght=length
    )

    # This is the total power losses vector
    powerLossesVector = resistanceVector * resultsCurrentVector**2
    # This are the total power losses
    powerLosses = np.sum(powerLossesVector)

    # Power losses per phase
    powPhA = np.sum(powerLossesVector[0:elementsPhaseA])
    powPhB = np.sum(
        powerLossesVector[elementsPhaseA : elementsPhaseA + elementsPhaseB : 1]
    )
    powPhC = np.sum(powerLossesVector[elementsPhaseA + elementsPhaseB :])

    # Results of power losses
    if not simple and not csv:
        print()
        print("----------------------------------------------------------------")
        print("Results of power losses")
        print(f"\tgeometry: {config['geometry']}")
        print(f"\tI={config['current']}[A], f={f}[Hz], l={length}[mm], T={t}[degC]")
        print("----------------------------------------------------------------")
        print(f"Sum [W]\t| dPa [W]\t| dPb [W]\t| dPc [W]")
        print(f"{powerLosses:.2f}\t| {powPhA:.2f} \t| {powPhB:.2f} \t| {powPhC:.2f}")
        print("----------------------------------------------------------------")
    elif not csv:
        print(f"{f}[Hz] \t {powerLosses:.2f} [W]")
    else:
        print(f"{f},{powerLosses:.2f}")

    if config["results"]:
        # getting the current density
        resultsCurrentVector *= 1 / (dXmm * dYmm)
        currentsDraw = csdm.recreateresultsArray(
            elementsVector, resultsCurrentVector, XSecArray
        )
        minCurrent = resultsCurrentVector.min()
        maxCurrent = resultsCurrentVector.max()

        base_cmap = plt.get_cmap("jet", 256)
        colors = base_cmap(np.arange(256))
        colors[0] = [1, 1, 1, 1]
        cmap = ListedColormap(colors)
        norm = plt.Normalize(vmin=0, vmax=maxCurrent)

        plt.imshow(currentsDraw, cmap=cmap, norm=norm)

        # Adjust the ticks
        ax = plt.gca()
        num_ticks_x = len(currentsDraw[0])
        num_ticks_y = len(currentsDraw)

        # Set the ticks and corresponding labels
        step = int(num_ticks_x / (num_ticks_x * dXmm / 10))

        x_ticks = np.arange(0, num_ticks_x, step)
        y_ticks = np.arange(0, num_ticks_y, step)

        # Set the ticks based on the array dimensions
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Set the tick labels by multiplying the tick values by the scaling factor
        ax.set_xticklabels((x_ticks * dXmm).astype(int))
        ax.set_yticklabels((y_ticks * dYmm).astype(int))

        # Add a color bar
        cbar = plt.colorbar()
        cbar.set_label("Current density [A/mm2]", rotation=270, labelpad=20)

        plt.title(
            f"I={config['current']}A, f={f}Hz, l={length}mm, Temp={t}degC\n\n\
total dP = {powerLosses:.2f}[W]\n\
dPa= {powPhA:.2f}[W] dPb= {powPhB:.2f}[W] dPc= {powPhC:.2f}[W]\n \n\
Current Density distribution [A/mm2]",
            fontsize=10,
            ha="center",
            pad=20,
        )

        plt.show()

# Doing the main work here.
if __name__ == "__main__":
    main()
