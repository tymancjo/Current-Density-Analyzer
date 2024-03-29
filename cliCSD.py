"""
This file is intended to be the Command Line Interface 
fot the CSD tool aimed to the quick analysis
for power losses in given geometry. 
The idea is to be able to use the saved geometry file
and deliver the required input as a command line
parameters. 

As an output the myLoged info of power losses 
is generated on the standard output. 
"""

# TODO:
# 1. Read the command line parameters - done
# 2. Loading the main geometry array from the file - done
# 3. Setup the solver - done
# 4. Solve - done
# 5. Prepare results - done
# 6. myLog results - done


# General imports
import numpy as np
import os.path
import sys

import pickle

# 1.
import argparse


# # Importing local library
# from csdlib import csdlib as csd


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

# use_njit = not True


# 2
def loadTheData(filename):
    """
    This is sub function to load data
    """

    if os.path.isfile(filename):
        myLog("reading from file :" + filename)
        XSecArray, dXmm, dYmm = loadObj(filename).restore()
        return XSecArray, dXmm, dYmm
    else:
        myLog(f"The file {filename} can't be opened!")
        sys.exit(1)


def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, args, kwargs)


def getArgs():
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
    parser.add_argument(
        "-sp", "--simple", action="store_true", help="Show only simple output"
    ),

    parser.add_argument(
        "-csv", "--csv", action="store_true", help="Show only simple output as csv f,dP"
    ),
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display the detailed information along process.",
    )

    parser.add_argument("geometry", help="Geometry description file in .csd format")
    parser.add_argument(
        "current",
        help="Current RMS value for the 3 phase symmetrical analysis in ampers [A]",
        type=float,
    )
    args = parser.parse_args()
    return vars(args)


def loadObj(filename):
    """load object data from file that was saved by saveObj function.
    Inputs:
    filename: file to save the data to with properly delivered path.

    Returns:
    Recreated object

    Example use:``

    P = loadObj('project.save')
    recreate P as myProject class object.
    """
    with open(filename, "rb") as myInput:
        return pickle.load(myInput)


def solveTheEquation(admitanceMatrix, voltageVector):
    return np.matmul(admitanceMatrix, voltageVector)


def getGmatrix(input):
    return np.linalg.inv(input)


# @njit
@conditional_decorator(njit, use_njit)
def N_getImpedanceArray(
    distanceArray,
    freq,
    dXmm,
    dYmm,
    lenght=1000,
    temperature=20,
    sigma20C=58e6,
    temCoRe=3.9e-3,
):
    """
    Calculate the array of impedance as complex values for each element
    Input:
    distanceArray -  array of distances beetween the elements in [mm]
    freq = frequency in Hz
    dXmm - size of element in x [mm]
    dYmm - size of element in Y [mm]
    lenght - analyzed lenght in [mm] /default= 1000mm
    temperature - temperature of the conductors in deg C / default = 20degC
    sigma20C - conductivity of conductor material in 20degC in [S] / default = 58MS (copper)
    temCoRe - temperature resistance coefficient / default is copper
    """
    omega = 2 * np.pi * freq

    impedanceArray = np.zeros((distanceArray.shape), dtype=np.complex_)

    for X in range(distanceArray.shape[0]):
        for Y in range(distanceArray.shape[0]):
            if X == Y:
                impedanceArray[Y, X] = N_getResistance(
                    sizeX=dXmm,
                    sizeY=dYmm,
                    lenght=lenght,
                    temp=temperature,
                    sigma20C=sigma20C,
                    temCoRe=temCoRe,
                ) + 1j * omega * N_getSelfInductance(
                    sizeX=dXmm, sizeY=dYmm, lenght=lenght
                )
            else:
                impedanceArray[Y, X] = (
                    1j
                    * omega
                    * N_getMutualInductance(
                        sizeX=dXmm,
                        sizeY=dYmm,
                        lenght=lenght,
                        distance=distanceArray[Y, X],
                    )
                )
    return impedanceArray


# Calculation of self inductance value function
# @njit
@conditional_decorator(njit, use_njit)
def N_getSelfInductance(sizeX, sizeY, lenght):
    """
    Calculate the self inductance for the subconductor
    It assumes rectangular shape. If you want put for circular shape just
    make sizeX = sizeY = 2r

    Inputs:
    sizeX - width in [mm]
    sizeY - height in [mm]
    lenght - cinductor lenght in [mm]

    output
    L in [H]
    """
    srednica = (sizeX + sizeY) / 2
    a = srednica * 1e-3
    l = lenght * 1e-3
    mi0 = 4 * np.pi * 1e-7

    # This calculation is based on the:
    # https://pdfs.semanticscholar.org/b0f4/eff92e31d4c5ff42af4a873ebdd826e610f5.pdf
    L = (mi0 * l / (2 * np.pi)) * (
        np.log(2 * l / a) - np.log(2) / 3 + 13 / 12 - np.pi / 2
    )

    # this was the previous formula
    # return 0.000000001*2*100*lenght*1e-3*(np.log(2*lenght*1e-3/(0.5*srednica*1e-3))-(3/4))

    return L


# @njit
@conditional_decorator(njit, use_njit)
def N_getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    """
    Calculate the resistance of the al'a square shape in given temperature
    All dimensions in mm
    temperature in deg C

    output:
    Resistance in Ohm
    """
    return (lenght / (sizeX * sizeY * sigma20C)) * 1e3 * (1 + temCoRe * (temp - 20))


# Calculations of mututal inductance between conductors
# @njit
@conditional_decorator(njit, use_njit)
def N_getMutualInductance(sizeX, sizeY, lenght, distance):
    """
    Calculate the mutual inductance for the subconductor
    It assumes rectangular shape. If you want put for circular shape just
    make sizeX = sizeY = 2r

    Inputs:
    sizeX - width in [mm]
    sizeY - height in [mm]
    lenght - conductor lenght in [mm]
    distance - distance between analyzed conductors in [mm]

    output
    M in [H]
    """
    srednica = (sizeX + sizeY) / 2

    a = srednica * 1e-3
    l = lenght * 1e-3
    d = distance * 1e-3
    mi0 = 4 * np.pi * 1e-7

    # fromula by:
    # https://pdfs.semanticscholar.org/b0f4/eff92e31d4c5ff42af4a873ebdd826e610f5.pdf
    M = (mi0 * l / (2 * np.pi)) * (
        np.log((l + np.sqrt(l**2 + d**2)) / d) - np.sqrt(1 + (d / l) ** 2) + d / l
    )

    # previous formula
    # return 0.000000001*2*lenght*1e-1*(np.log(2*lenght*1e-1/(distance/10))-(3/4))
    return M


# @njit
@conditional_decorator(njit, use_njit)
def N_getResistanceArray(
    elementsVector,
    dXmm,
    dYmm,
    lenght=1000,
    temperature=20,
    sigma20C=58e6,
    temCoRe=3.9e-3,
):
    """
    Calculate the array of resistance values for each element
    Input:
    elementsVector - The elements vector as delivered by arrayVectorize
    dXmm - size of element in x [mm]
    dYmm - size of element in Y [mm]
    lenght - analyzed lenght in [mm] /default= 1000mm
    temperature - temperature of the conductors in deg C / defoult = 20degC
    sigma20C - conductivity of conductor material in 20degC in [S] / default = 58MS (copper)
    temCoRe - temperature resistance coeficcient / default is copper
    """

    resistanceArray = np.zeros(elementsVector.shape[0])
    for element in range(elementsVector.shape[0]):
        resistanceArray[element] = N_getResistance(
            sizeX=dXmm,
            sizeY=dYmm,
            lenght=lenght,
            temp=temperature,
            sigma20C=sigma20C,
            temCoRe=temCoRe,
        )

    return resistanceArray


# Calculate distance between elements function
# @njit
@conditional_decorator(njit, use_njit)
def N_getDistancesArray(inputVector):
    """
    This function calculate the array of distances between every conductors
    element
    Input:
    the vector of conductor elements as delivered by n_vectorizeTheArray
    """
    # lets check for the numbers of elements
    elements = inputVector.shape[0]
    # print(elements)
    # Define the outpur array
    distanceArray = np.zeros((elements, elements))

    for x in range(elements):
        for y in range(elements):
            if x != y:
                posXa = inputVector[y][2]
                posYa = inputVector[y][3]

                posXb = inputVector[x][2]
                posYb = inputVector[x][3]

                distanceArray[y, x] = np.sqrt(
                    (posXa - posXb) ** 2 + (posYa - posYb) ** 2
                )
            else:
                distanceArray[y, x] = 0
    return distanceArray


# Master Array Vectorization FUNCTION
# @njit
@conditional_decorator(njit, use_njit)
def N_arrayVectorize(inputArray, phaseNumber, dXmm, dYmm):
    """
    Desription:
    This function returns vector of 4 dimension vectors that deliver

    input:
    inputArray = 3D array thet describe by 1's position of
    conductors in cross section
    dXmm - size of each element in X direction [mm]
    dYmm - size of each element in Y direction [mm]
    Output:
    [0,1,2,3] - 4 elements vector for each element, where:

    0 - Original inputArray geometry origin Row for the set cell
    1 - Original inputArray geometry origin Col for the set cell
    2 - X position in mm of the current element
    3 - Y position in mm of the current element

    Number of such [0,1,2,3] elements is equal to the number of defined
    conductor cells in geometry

    """
    # Let's check the size of the array
    elementsInY = inputArray.shape[0]
    elementsInX = inputArray.shape[1]

    # lets define the empty vectorArray
    vectorArray = []

    # lets go for each input array position and check if is set
    # and if yes then put it into putput vectorArray
    for Row in range(elementsInY):
        for Col in range(elementsInX):
            if inputArray[Row][Col] == phaseNumber:
                # Let's calculate the X and Y coordinates
                coordinateY = (0.5 + Row) * dYmm
                coordinateX = (0.5 + Col) * dXmm

                vectorArray.append([Row, Col, coordinateX, coordinateY])

    return np.array(vectorArray)


# @njit
# @conditional_decorator(njit, use_njit)
def N_arraySlicer(inputArray, subDivisions=2):
    """
    This function increase the resolution of the cross section array
    inputArray -  oryginal geometry matrix
    subDivisions -  number of subdivisions / factor of increase of resoluttion / default = 2
    """
    return inputArray.repeat(subDivisions, axis=0).repeat(subDivisions, axis=1)


def N_getComplexModule(x):
    """
    returns the module of complex number
    input: x - complex number
    if not a complex number is given as parameter then it return the x diretly

    """
    if isinstance(x, complex):
        return np.absolute(x)
        # return np.sqrt(x.real ** 2 + x.imag ** 2)
    else:
        return x


# Doing the main work here.
if __name__ == "__main__":
    config = getArgs()
    verbose = config["verbose"]
    simple = config["simple"]
    csv = config["csv"]

    myLog()
    myLog("Starting operations...")
    myLog()

    # 2 loading the geometry data:
    XSecArray, dXmm, dYmm = loadTheData(config["geometry"])
    myLog()
    myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    myLog(f"Data table size: {XSecArray.shape}")

    if config["split"] > 1:
        myLog()
        myLog("Splitting the geometry cells...", end="")
        splits = 1
        for _ in range(config["split"] - 1):
            if dXmm > 1 and dYmm > 1:
                myLog(f"{splits}... ", end="")
                splits += 1
                XSecArray = N_arraySlicer(inputArray=XSecArray, subDivisions=2)
                dXmm = dXmm / 2
                dYmm = dYmm / 2
            else:
                myLog()
                myLog("No further subdivisions make sense")
                break

        myLog()
        myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
        myLog(f"Data table size: {XSecArray.shape}")

    # 3 preparing the solution
    Irms = config["current"]
    # Current vector
    I = [Irms, 0, Irms, 120, Irms, 240]
    f = config["frequency"]
    length = config["length"]
    t = config["Temperature"]

    myLog()
    myLog("Starting solver for")
    for k, n in zip([0, 2, 4], ["a", "b", "c"]):
        myLog(f"I{n} = {I[k]}[A] \t {I[k+1]}[deg] \t {f}[Hz]")

    myLog()
    myLog("Complex form:")

    # lets workout the  current in phases as is defined
    in_Ia = I[0] * np.cos(I[1] * np.pi / 180) + I[0] * np.sin(I[1] * np.pi / 180) * 1j
    myLog(f"Ia: {in_Ia}")

    in_Ib = I[2] * np.cos(I[3] * np.pi / 180) + I[2] * np.sin(I[3] * np.pi / 180) * 1j
    myLog(f"Ib: {in_Ib}")

    in_Ic = I[4] * np.cos(I[5] * np.pi / 180) + I[4] * np.sin(I[5] * np.pi / 180) * 1j
    myLog(f"Ic: {in_Ic}")

    vPhA = N_arrayVectorize(inputArray=XSecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm)
    vPhB = N_arrayVectorize(inputArray=XSecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm)
    vPhC = N_arrayVectorize(inputArray=XSecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm)

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
        myLog()
        myLog(
            "!!! Size of the elements vector may lead to very long calculation. Be aware!"
        )
        myLog("You can break the process by CTRL+C")
        myLog("You may conceder reduce the split steps.")
        myLog("Optimal element size is around 1.5x1.5mm")
        myLog()

    if len(elementsVector) > 10000:
        myLog("Extreme size of elements vector - long calculations immanent!")

    admitanceMatrix = getGmatrix(
        N_getImpedanceArray(
            N_getDistancesArray(elementsVector),
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
    currentVector = solveTheEquation(admitanceMatrix, voltageVector)

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

    myLog()
    myLog("Calculated require Source Voltages")
    myLog(Ua)
    myLog(Ub)
    myLog(Uc)

    # Setting up the voltage vector for final solve
    vA = np.ones(elementsPhaseA) * Ua
    vB = np.ones(elementsPhaseB) * Ub
    vC = np.ones(elementsPhaseC) * Uc

    voltageVector = np.concatenate((vA, vB, vC), axis=0)

    # Final solve
    # Main equation solve
    currentVector = solveTheEquation(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringing each phase current to the assumer Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)
    # end of second solve!

    myLog()
    myLog("Solution check...")
    myLog("Raw Current results:")
    myLog(f"Ia: {Ia}")
    myLog(f"Ib: {Ib}")
    myLog(f"Ic: {Ic}")
    myLog()
    myLog(f"Sum: {Ia+Ib+Ic}")

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

    myLog("Fix Current results:")
    myLog(f"Ia: {Ia}")
    myLog(f"Ib: {Ib}")
    myLog(f"Ic: {Ic}")
    myLog()
    myLog(f"Sum: {Ia+Ib+Ic}")

    # Data postprocessing
    getMod = np.vectorize(N_getComplexModule)

    resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)
    # for debug
    # myLog(resultsCurrentVector)
    #
    resultsCurrentVector = getMod(resultsCurrentVector)
    resistanceVector = N_getResistanceArray(
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
        print("------------------------------------------------------")
        print("Results of power losses")
        print(f"\tgeometry: {config['geometry']}")
        print(f"\tI={config['current']}[A], f={f}[Hz], l={length}[mm]")
        print("------------------------------------------------------")
        print(f"Sum [W]\t| dPa [W]\t| dPb [W]\t| dPc [W]")
        print(f"{powerLosses:.2f}\t| {powPhA:.2f} \t| {powPhB:.2f} \t| {powPhC:.2f}")
        print("------------------------------------------------------")
    elif not csv:
        print(f"{f}[Hz] \t {powerLosses:.2f} [W]")
    else:
        print(f"{f},{powerLosses:.2f}")
