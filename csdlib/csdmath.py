"""
this is the library of function to cover the math and calculation operations
in the CSD package. 
"""

import numpy as np

try:
    from numba import njit
    use_njit = True

except ImportError:
    use_njit = False
    njit = None

# making the NUMBA decorators optional
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator



# global constant to speed up loops
mi0 = 4 * np.pi * 1e-7
C1 = -np.log(2) / 3 + 13 / 12 - np.pi / 2
C2 = mi0 / (2 * np.pi)

# the functions
def solveTheEquation(admitanceMatrix, voltageVector):
    return np.matmul(admitanceMatrix, voltageVector)


def getGmatrix(input):
    return np.linalg.inv(input)


@conditional_decorator(njit, use_njit)
def getImpedanceArray(
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

    impedanceArray = np.zeros((distanceArray.shape), dtype=np.complex128)

    for X in range(distanceArray.shape[0]):
        for Y in range(distanceArray.shape[0]):
            if X == Y:
                impedanceArray[Y, X] = getResistance(
                    sizeX=dXmm,
                    sizeY=dYmm,
                    lenght=lenght,
                    temp=temperature,
                    sigma20C=sigma20C,
                    temCoRe=temCoRe,
                ) + 1j * omega * getSelfInductance(
                    sizeX=dXmm, sizeY=dYmm, lenght=lenght
                )
            else:
                impedanceArray[Y, X] = (
                    1j
                    * omega
                    * getMutualInductance(
                        sizeX=dXmm,
                        sizeY=dYmm,
                        lenght=lenght,
                        distance=distanceArray[Y, X],
                    )
                )
    return impedanceArray



@conditional_decorator(njit, use_njit)
def getSelfInductance(sizeX, sizeY, lenght):
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

    # This calculation is based on the:
    # https://pdfs.semanticscholar.org/b0f4/eff92e31d4c5ff42af4a873ebdd826e610f5.pdf
    # L = (mi0 * l / (2 * np.pi)) * (
    #     np.log(2 * l / a) - np.log(2) / 3 + 13 / 12 - np.pi / 2
    # )

    # this is the above formula just with pre calculated constant
    L = (C2 * l) * (np.log(2 * l / a) + C1)

    # this was the previous formula
    # return 0.000000001*2*100*lenght*1e-3*(np.log(2*lenght*1e-3/(0.5*srednica*1e-3))-(3/4))

    return L



@conditional_decorator(njit, use_njit)
def getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    """
    Calculate the resistance of the al'a square shape in given temperature
    All dimensions in mm
    temperature in deg C

    output:
    Resistance in Ohm
    """
    return (lenght / (sizeX * sizeY * sigma20C)) * 1e3 * (1 + temCoRe * (temp - 20))



@conditional_decorator(njit, use_njit)
def getMutualInductance(sizeX, sizeY, lenght, distance):
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
    # mi0 = 4 * np.pi * 1e-7

    # formula by:
    # https://pdfs.semanticscholar.org/b0f4/eff92e31d4c5ff42af4a873ebdd826e610f5.pdf
    # M = (mi0 * l / (2 * np.pi)) * (
    #     np.log((l + np.sqrt(l**2 + d**2)) / d) - np.sqrt(1 + (d / l) ** 2) + d / l
    # )
    # the same as above with the pre-calculated C2
    M = (
        C2
        * l
        * (np.log((l + np.sqrt(l**2 + d**2)) / d) - np.sqrt(1 + (d / l) ** 2) + d / l)
    )

    # previous formula
    # return 0.000000001*2*lenght*1e-1*(np.log(2*lenght*1e-1/(distance/10))-(3/4))
    return M



@conditional_decorator(njit, use_njit)
def getResistanceArray(
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
        resistanceArray[element] = getResistance(
            sizeX=dXmm,
            sizeY=dYmm,
            lenght=lenght,
            temp=temperature,
            sigma20C=sigma20C,
            temCoRe=temCoRe,
        )

    return resistanceArray


@conditional_decorator(njit, use_njit)
def getDistancesArray(inputVector):
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



def arrayVectorize(inputArray, phaseNumber, dXmm, dYmm):
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


def arraySlicer(inputArray, subDivisions=2):
    """
    This function increase the resolution of the cross section array
    inputArray -  oryginal geometry matrix
    subDivisions -  number of subdivisions / factor of increase of resoluttion / default = 2
    """
    return inputArray.repeat(subDivisions, axis=0).repeat(subDivisions, axis=1)


def getComplexModule(x):
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


# Function that put back together the solution vectr back to represent the cross
# section shape array
def recreateresultsArray(elementsVector, resultsVector, initialGeometryArray):
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
