"""
This file is intended to be the Command Line Interface 
fot the CSD tool aimed to the quick analysis
for power losses in given geometry. 
The idea is to be able to use the saved geometry file
and deliver the required input as a command line
parameters. 

As an output the printed info of power losses 
is generated on the standard output. 
"""

# TODO:
# 1. Read the command line parameters - done
# 2. Loading the main geometry array from the file - done
# 3. Setup the solver - done
# 4. Solve - done
# 5. Prepare results - done
# 6. Print results - done


# General imports
import numpy as np
import os.path
import sys

# 1.
import argparse


# Importing local library
from csdlib import csdlib as csd


# 2
def loadTheData(filename):
    """
    This is sub function to load data
    """

    if os.path.isfile(filename):
        print("reading from file :" + filename)
        XSecArray, dXmm, dYmm = csd.loadObj(filename).restore()
        return XSecArray, dXmm, dYmm
    else:
        print(f"The file {filename} can't be opened!")
        sys.exit(1)


# Doing the main work here.
if __name__ == "__main__":

    # 1 handling the in line parameters
    parser = argparse.ArgumentParser(
        description="CSD cli executor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--split", help="Split geometry steps", type=int, default=1
    )
    parser.add_argument("-f", "--frequency", type=float, default=50.0)
    parser.add_argument("-T", "--Temperature", type=float, default=140.0)
    parser.add_argument("-l", "--length", type=float, default=1000.0)

    parser.add_argument("geometry", help="Geometry description file in .csd format")
    parser.add_argument(
        "current",
        help="Current RMS value for the 3 phase symmetrical analysis in ampers [A]",
        type=float,
    )
    args = parser.parse_args()
    config = vars(args)
    # print(config)

    print()
    print("Starting operations...")
    print()

    # 2 loading the geometry data:
    XSecArray, dXmm, dYmm = loadTheData(config["geometry"])
    print()
    print(f"dX:{dXmm}mm dY:{dYmm}mm")
    print(f"Data table size: {XSecArray.shape}")

    if config["split"] > 1:
        print()
        print("Splitting the geometry cells...", end="")
        splits = 1
        for _ in range(config["split"] - 1):
            if dXmm > 1 and dYmm > 1:
                print(f"{splits}... ", end="")
                splits += 1
                XSecArray = csd.n_arraySlicer(inputArray=XSecArray, subDivisions=2)
                dXmm = dXmm / 2
                dYmm = dYmm / 2
            else:
                print()
                print("No further subdivisions make sense")
                break

        print()
        print(f"dX:{dXmm}mm dY:{dYmm}mm")
        print(f"Data table size: {XSecArray.shape}")

    # 3 preparing the solution
    Irms = config["current"]
    # Current vector
    I = [Irms, 0, Irms, 120, Irms, 240]
    f = config["frequency"]
    length = config["length"]
    t = config["Temperature"]

    print()
    print("Starting solver for")
    for k, n in zip([0, 2, 4], ["a", "b", "c"]):
        print(f"I{n} = {I[k]}[A] \t {I[k+1]}[deg] \t {f}[Hz]")

    print()
    print("Complex form:")

    # lets workout the  current in phases as is defined
    in_Ia = I[0] * np.cos(I[1] * np.pi / 180) + I[0] * np.sin(I[1] * np.pi / 180) * 1j
    print(f"Ia: {in_Ia}")

    in_Ib = I[2] * np.cos(I[3] * np.pi / 180) + I[2] * np.sin(I[3] * np.pi / 180) * 1j
    print(f"Ib: {in_Ib}")

    in_Ic = I[4] * np.cos(I[5] * np.pi / 180) + I[4] * np.sin(I[5] * np.pi / 180) * 1j
    print(f"Ic: {in_Ic}")

    vPhA = csd.n_arrayVectorize(
        inputArray=XSecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm
    )
    vPhB = csd.n_arrayVectorize(
        inputArray=XSecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm
    )
    vPhC = csd.n_arrayVectorize(
        inputArray=XSecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm
    )

    # Lets put the all phases together
    elementsPhaseA = len(vPhA)
    elementsPhaseB = len(vPhB)
    elementsPhaseC = len(vPhC)

    if elementsPhaseA != 0 and elementsPhaseB != 0 and elementsPhaseC != 0:
        elementsVector = np.concatenate((vPhA, vPhB, vPhC), axis=0)
    elif elementsPhaseA == 0:
        if elementsPhaseB == 0:
            elementsVector = vPhC
        elif elementsPhaseC == 0:
            elementsVector = vPhB
        else:
            elementsVector = np.concatenate((vPhB, vPhC), axis=0)
    else:
        if elementsPhaseB == 0 and elementsPhaseC == 0:
            elementsVector = vPhA
        elif elementsPhaseC == 0:
            elementsVector = np.concatenate((vPhA, vPhB), axis=0)
        else:
            elementsVector = np.concatenate((vPhA, vPhC), axis=0)

    if len(elementsVector) > 1200:
        print()
        print(
            "!!! Size of the elements vector may lead to very long calculation. Be aware!"
        )
        print("You can break the process by CTRL+C")
        print("You may conceder reduce the split steps.")
        print("Optimal element size is around 1.5x1.5mm")
        print()

    if len(elementsVector) > 10000:
        print("Extreme size of elements vector - long calculations immanent!")

    admitanceMatrix = np.linalg.inv(
        csd.n_getImpedanceArray(
            csd.n_getDistancesArray(elementsVector),
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

    voltageVector = np.concatenate((vA, vB, vC), axis=0)

    # Initial solve
    # Main equation solve
    currentVector = np.matmul(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringin each phase current to the assumer Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)

    # expected Ia Ib Ic as symmetrical ones
    # ratios of currents will give us new voltages for phases
    Ua = Ua * (in_Ia / Ia)
    Ub = Ub * (in_Ib / Ib)
    Uc = Uc * (in_Ic / Ic)

    print()
    print("Calculated require Source Voltages")
    print(Ua)
    print(Ub)
    print(Uc)

    # Setting up the voltage vector for final solve
    vA = np.ones(elementsPhaseA) * Ua
    vB = np.ones(elementsPhaseB) * Ub
    vC = np.ones(elementsPhaseC) * Uc

    voltageVector = np.concatenate((vA, vB, vC), axis=0)

    # Final solve
    # Main equation solve
    currentVector = np.matmul(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringing each phase current to the assumer Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)
    # end of second solve!

    print()
    print("Solution check...")
    print("Raw Current results:")
    print(f"Ia: {Ia}")
    print(f"Ib: {Ib}")
    print(f"Ic: {Ic}")
    print()
    print(f"Sum: {Ia+Ib+Ic}")

    # Now we normalize up to the expecter I - just a polish
    # as we are almost there with the previous second solve for new VOLTAGES
    modIa = np.abs(Ia)
    modIb = np.abs(Ib)
    modIc = np.abs(Ic)

    currentPhA *= in_Ia / modIa
    currentPhB *= in_Ib / modIb
    currentPhC *= in_Ic / modIc

    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)

    print("Fix Current results:")
    print(f"Ia: {Ia}")
    print(f"Ib: {Ib}")
    print(f"Ic: {Ic}")
    print()
    print(f"Sum: {Ia+Ib+Ic}")

    # Data postprocessing
    getMod = np.vectorize(csd.n_getComplexModule)

    resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)
    # for debug
    # print(resultsCurrentVector)
    #
    resultsCurrentVector = getMod(resultsCurrentVector)
    resistanceVector = csd.n_getResistanceArray(
        elementsVector, dXmm=dXmm, dYmm=dYmm, temperature=t, lenght=length
    )

    # This is the total power losses vector
    powerLossesVector = resistanceVector * resultsCurrentVector ** 2
    # This are the total power losses
    powerLosses = np.sum(powerLossesVector)

    # Power losses per phase
    powPhA = np.sum(powerLossesVector[0:elementsPhaseA])
    powPhB = np.sum(
        powerLossesVector[elementsPhaseA : elementsPhaseA + elementsPhaseB : 1]
    )
    powPhC = np.sum(powerLossesVector[elementsPhaseA + elementsPhaseB :])

    # Results of power losses
    print()
    print("---------------------------")
    print("Results of power losses")
    print(f"for I={config['current']}[A], f={f}[Hz], l={length}[mm]")
    print(f"Total 3 phase power losses: \t {powerLosses:.2f}[W]")
    print("---------------------------")
    print("Per phase power losses:")
    print(f"for I={config['current']}[A], f={f}[Hz], l={length}[mm]")
    print(f"dPa: {powPhA:.2f} \t dPb: {powPhB:.2f} \t dPc: {powPhC:.2f}")
    print("---------------------------")
