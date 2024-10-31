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
# 7. adding inner code working - done

# General imports
import numpy as np
import os.path
import sys

import pickle

# 1.
import argparse


# # Importing local library
# from csdlib import csdlib as csd
from csdlib import innercode as ic 


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
    njit = None

# use_njit = not True


# 2
def loadTheData(filename):
    """
    This is sub function to load data
    """

    if os.path.isfile(filename):
        _, extension = os.path.splitext(filename)
        myLog("File type: " + extension)

        if extension.lower() in [".txt", ".inc", ".ic"]:
            myLog("reading the inner-code geometry file: " + filename)
            try:
                with open(filename, "r") as f:
                    file_content = f.read()

                XSecArray, dXmm, dYmm = getCanvas(file_content)
            except IOError:
                print("Error reading the file " + filename)
                sys.exit(1)

        else:
            myLog("reading from file :" + filename)
            XSecArray, dXmm, dYmm = loadObj(filename).restore()

        return XSecArray, dXmm, dYmm
    else:
        myLog(f"The file {filename} can't be opened!")
        sys.exit(1)


def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, *args, *kwargs)


def getArgs():
    # 1 handling the in line parameters
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


# ### General geometry generators ###
# def addCircle(x0, y0, D1, Set, D2=0, Set2=0, draw=True, shift=(0, 0)):
#     """Generalized formula to add circle at given position (x,y) [mm]
#     of a two diameters external D1 and internal D2 (if a donat is needed) [mm]"""

#     if draw:
#         # this works on global canvas array
#         global XSecArray

#         x0 = x0 - shift[0]
#         y0 = y0 - shift[1]

#         r1sq = (D1 / 2) ** 2
#         r2sq = (D2 / 2) ** 2

#         elementsInY = XSecArray.shape[0]
#         elementsInX = XSecArray.shape[1]

#         for x in range(elementsInX):
#             for y in range(elementsInY):
#                 xmm = x * dXmm + dXmm / 2
#                 ymm = y * dXmm + dXmm / 2
#                 distSq = (xmm - x0) ** 2 + (ymm - y0) ** 2
#                 if distSq < r2sq:
#                     XSecArray[y, x] = Set2
#                 elif distSq <= r1sq:
#                     XSecArray[y, x] = Set

#     x0 = x0 - D1 / 2
#     y0 = y0 - D1 / 2
#     xE = x0 + D1
#     yE = y0 + D1
#     return [x0, y0, xE, yE]


# def addRect(x0, y0, W, H, Set, draw=True, shift=(0, 0)):
#     """Generalized formula to add rectangle at given position
#     start - left top corner(x,y)[mm]
#     width, height[mm]"""

#     xE = x0 + W
#     yE = y0 + H

#     if draw:
#         # this works on global canvas array
#         global XSecArray
#         x0 = x0 - shift[0]
#         y0 = y0 - shift[1]
#         xE = x0 + W
#         yE = y0 + H

#         elementsInY = XSecArray.shape[0]
#         elementsInX = XSecArray.shape[1]

#         for x in range(elementsInX):
#             for y in range(elementsInY):
#                 xmm = x * dXmm + dXmm / 2
#                 ymm = y * dXmm + dXmm / 2

#                 if (x0 <= xmm <= xE) and (y0 <= ymm <= yE):
#                     XSecArray[y, x] = Set

#     return [x0, y0, xE, yE]


# def textToCode(input_text):
#     """This is the function that will return the list 
#     of geometry execution code stps.
#     Code commands are in the form of dictionary"""

#     commands = {
#         'c': [addCircle,[4,6]],
#         'r': [addRect,[5]],
#         'v': [None,[2]],
#         'a': [None, [2]],
#         'l': [None,[1]]
#     }

#     innerCodeSteps = []
#     innerVariables = {}
    

#     for line_nr,line in enumerate(input_text):
#         if len(line)>3:
#             command = line[0].lower()
#             if command in commands:
#                 if command == 'l':
#                     # taking care of looping the stuff
#                     ar = line[2:-1].split(',')
#                     if len(ar) in commands[command][1]:
#                         loops = int(ar[0])
#                         loop_code = input_text[line_nr+1:]
#                         for _ in range(loops):
#                             input_text.extend(loop_code)

#     for line_nr,line in enumerate(input_text):
#         if len(line)>5:
#             command = line[0].lower()
#             if command in commands:
#                 if command == 'v':
#                     # taking care if the command sets the variable
#                     ar = line[2:-1].split(',')
#                     if len(ar) in commands[command][1]:
#                         variable_name = str(ar[0])
#                         variable_value = float(ar[1])
#                         innerVariables[variable_name] = variable_value
#                 elif command == 'a':
#                     if len(innerVariables):
#                         # taking care if the command sets the variable
#                         ar = line[2:-1].split(',')
#                         if len(ar) in commands[command][1]:
#                             variable_name = str(ar[0])
#                             variable_value = innerVariables[variable_name]+float(ar[1])
#                             innerVariables[variable_name] = variable_value

#                 else:
#                     # ar as argumnents 
#                     ar = line[2:-1].split(',')
#                     # insert inner variables if any
#                     if len(innerVariables):
#                         # let's replace the variables with values
#                         for i,argument in enumerate(ar):
#                             if argument in innerVariables:
#                                 ar[i] = innerVariables[argument]

#                     if len(ar) in commands[command][1]:
#                         ar = [float(a) for a in ar]
#                         innerCodeSteps.append([commands[command][0],ar,command])

#     return innerCodeSteps


def getCanvas(textInput):
    """This function is to determine the best parameters for the canvas
    based on the given geometry steps defined by the inner code."""

    global XSecArray, dXmm, dYmm

    codeLines = textInput.splitlines()
    codeSteps = ic.textToCode(codeLines)

    X = []
    Y = []

    circles = False
    if codeSteps:
        for step in codeSteps:
            tmp = step[0](*step[1], draw=False)
            if step[0] is ic.addCircle:
                circles = True

            X.append(tmp[0])
            X.append(tmp[2])
            Y.append(tmp[1])
            Y.append(tmp[3])

        myLog(X)
        myLog(Y)
        myLog(f"Dimention range: {min(X)}:{max(X)}; {min(Y)}:{max(Y)}")
        size = (max(X) - min(X), max(Y) - min(Y))
        myLog(size)

        # I have no good idea how to figure out the best cell size
        # so for now it's just some stuff..
        if circles:
            sizes = [4, 2.5, 2, 1]
        else:
            sizes = [15, 10, 5, 4, 2.5, 2, 1]
        for xd in sizes:
            if (size[0] % xd == 0) and (size[1] % xd == 0):
                break
        myLog(f"The dx: {xd}mm")

        elements_x = int(size[1] / xd)
        elements_y = int(size[0] / xd)

        myLog(f"Canvas elements need: {elements_x, elements_y}")

        dXmm = dYmm = xd
        XSecArray = np.zeros([elements_x, elements_y])

        for step in codeSteps:
            step[0](*step[1], shift=(min(X), min(Y)),XSecArray=XSecArray,dXmm=dXmm)
        return XSecArray, dXmm, dYmm

    return False


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


# Function that put back together the solution vectr back to represent the crss section shape array
def N_recreateresultsArray(elementsVector, resultsVector, initialGeometryArray):
    """
    Functions returns recreate cross section array with mapperd solution results
    Inputs:
    elementsVector - vector of crossection elements as created by the n_arrayVectorize
    resultsVector - vectr with results values calculated base on the elementsVector
    initialGeometryArray - the array that contains the cross section geometry model
    """
    localResultsArray = np.zeros((initialGeometryArray.shape), dtype=float)

    for vectorIndex, result in enumerate(resultsVector):
        localResultsArray[
            int(elementsVector[vectorIndex][0]), int(elementsVector[vectorIndex][1])
        ] = result

    return localResultsArray


# Doing the main work here.
if __name__ == "__main__":
    config = getArgs()
    verbose = config["verbose"]
    simple = config["simple"]
    csv = config["csv"]

    myLog()
    myLog("Starting operations...")
    myLog()

    if config["draw"] or config["results"]:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

    XSecArray = np.zeros((0, 0))
    dXmm = dYmm = 1

    # 2 loading the geometry data:
    XSecArray, dXmm, dYmm = loadTheData(config["geometry"])
    myLog("Initial geometry array parameters:")
    myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    myLog(f"Data table size: {XSecArray.shape}")

    
    while dXmm > config["size"]:
        myLog("Splitting the geometry cells...", end="")
        XSecArray = N_arraySlicer(inputArray=XSecArray, subDivisions=2)
        dXmm = dXmm / 2
        dYmm = dYmm / 2
    
    myLog()
    myLog("Adjusted geometry array parameters:")
    myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    myLog(f"Data table size: {XSecArray.shape}")


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

    if config["results"]:
        # getting the current density
        resultsCurrentVector *= 1 / (dXmm * dYmm)
        currentsDraw = N_recreateresultsArray(
            elementsVector, resultsCurrentVector, XSecArray
        )
        minCurrent = resultsCurrentVector.min()
        maxCurrent = resultsCurrentVector.max()

        base_cmap = plt.cm.get_cmap("jet", 256)
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
