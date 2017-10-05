'''
This is separate library file for the CSD.py applications
Its done this way to clean up the code in the main app

v00 - Initial Build
'''

### External Loads
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import pickle

from tkinter import *
from tkinter import filedialog, messagebox

import functools
import numpy as np

import os.path

from csdlib.vect import Vector as v2

### End of external Loads
# Classes

class cointainer():
    def __init__(self, xArray, dX, dY):
        self.xArray = xArray
        self.dX = dX
        self.dY = dY
        print('The cointainer is created')

    def save(self, filename):
        filename = str(filename) + '.csd'
        try:
            print('trying to save to: {}'.format(filename))
            tempfile = filename + '_temp'
            with open(tempfile, 'wb') as output:
                pickle.dump(self, output, pickle.DEFAULT_PROTOCOL)
        except:
            print('There is issue with pickle. Save aborded to protect file.')
        else:
            with open(filename, 'wb') as output:
                pickle.dump(self, output, pickle.DEFAULT_PROTOCOL)

    def restore(self):
        return self.xArray, self.dX, self.dY

# ################# FUNCTIONS & PROCEDURES##############################


def loadObj(filename):
    '''load object data from file that was saved by saveObj function.
    Inputs:
    filename: file to save the data to with properly delivered path.

    Returns:
    Recreated object

    Example use:``

    P = loadObj('project.save')
    recreate P as myProject class object.
    '''
    with open(filename, 'rb') as myInput:
        return pickle.load(myInput)


# Calculations of mututal inductance between conductors
def n_getMutualInductance(sizeX, sizeY, lenght, distance):
    '''
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
    '''
    srednica = (sizeX+sizeY)/2

    return 0.000000001*2*lenght*1e-1*(np.log(2*lenght*1e-1/(distance/10))-(3/4))



# Calculation of self inductance value function
def n_getSelfInductance(sizeX, sizeY, lenght):
    '''
    Calculate the self inductance for the subconductor
    It assumes rectangular shape. If you want put for circular shape just
    make sizeX = sizeY = 2r

    Inputs:
    sizeX - width in [mm]
    sizeY - height in [mm]
    lenght - cinductor lenght in [mm]

    output
    L in [H]
    '''
    srednica = (sizeX+sizeY)/2
    return 0.000000001*2*100*lenght*1e-3*(np.log(2*lenght*1e-3/(0.5*srednica*1e-3))-(3/4))


# Calculate the resistance value function
def n_getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    '''
    Calculate the resistance of the al'a square shape in given temperature
    All dimensions in mm
    temperature in deg C

    output:
    Resistance in Ohm
    '''
    return (lenght/(sizeX*sizeY*sigma20C)) * 1e3 * (1+temCoRe*(temp-20))


# Calculate distance between elements function
def n_getDistancesArray(inputVector):
    '''
    This function calculate the array of distances between every conductors
    element
    Input:
    the vector of conductor elements as delivered by n_vectorizeTheArray
    '''
    # lets check for the numbers of elements
    elements = inputVector.shape[0]
    print(elements)
    # Define the outpur array
    distanceArray = np.zeros((elements, elements))

    for x in range(elements):
        for y in range(elements):
            if x != y:
                posXa = inputVector[y][2]
                posYa = inputVector[y][3]

                posXb = inputVector[x][2]
                posYb = inputVector[x][3]

                distanceArray[y, x] = np.sqrt((posXa-posXb)**2 + (posYa-posYb)**2)
            else:
                distanceArray[y, x] = 0
    return distanceArray


def n_perymiter(vec, arr, dXmm, dYmm):
    '''
    This function returns the area perynmiter lenght for given
    vector of conducting elements in the array
    Inputs:
    vec - vector of elements to calculate the perymiter
        lenght for (as delivered by n_vectorizeTheArray)

    arr - array that describe the geometry shape

    dXmm - element size in x diretion

    dYmm - element size in y diretion

    Output:
    perymiter lenght in the same units as dXmm and dYmm
    '''
    # TODO: adding check if we dont exeed dimensions of array
    perymiter = 0
    for box in vec:

        # checking in x diretions lef and right
        if arr[int(box[0] + 1)][int(box[1])] == 0:
            perymiter += dYmm
        if arr[int(box[0] - 1)][int(box[1])] == 0:
            perymiter += dYmm

        if arr[int(box[0])][int(box[1] + 1)] == 0:
            perymiter += dXmm
        if arr[int(box[0])][int(box[1] - 1)] == 0:
            perymiter += dXmm

    return perymiter


# Master Array Vecrorization FUNCTION
def n_arrayVectorize(inputArray, phaseNumber, dXmm, dYmm):
    '''
    Desription:
    This function returns vector of 4 dimension vectors that deliver

    input:
    inputArray = 3D array thet describe by 1's position of
    conductors in cross section
    dXmm - size of each element in X direction [mm]
    dYmm - size of each element in Y diretion [mm]
    Output:
    [0,1,2,3] - 4 elemets vector for each element, where:

    0 - Oryginal inputArray geometry origin Row for the set cell
    1 - Oryginal inputArray geometry origin Col for the set cell
    2 - X position in mm of the current element
    3 - Y position in mm of the current element

    Number of such [0,1,2,3] elements is equal to the number of defined
    conductor cells in geometry

    '''
    # Let's check the size of the array
    elementsInY = inputArray.shape[0]
    elementsInX = inputArray.shape[1]

    #lets define the empty vectorArray
    vectorArray = []

    #lets go for each input array position and check if is set
    #and if yes then put it into putput vectorArray
    for Row in range(elementsInY):
        for Col in range(elementsInX):
            if inputArray[Row][Col] == phaseNumber:
                # Let's calculate the X and Y coordinates
                coordinateY = (0.5 + Row) * dYmm
                coordinateX = (0.5 + Col) * dXmm

                vectorArray.append([Row, Col, coordinateX, coordinateY])

    return np.array(vectorArray)

# Canvas preparation procedure
def n_checkered(canvas, cutsX, cutsY):
    '''
    This function clean the board and draw grid
    Inputs:
    canvas - tkinter canvas object
    cutsX - elements in X (left right) direction
    cutsY - elements in Y (top down) direction
    '''

    # Reading the size of the canvas element
    canvasHeight = canvas.winfo_height()
    canvasWidth  = canvas.winfo_width()

    line_distanceX = (canvasWidth / cutsX)
    line_distanceY = (canvasHeight / cutsY)


    # Cleaning up the whole canvas space by drawing a white rectangle
    canvas.create_rectangle(0, 0, canvasWidth, canvasHeight, fill="white", outline="gray")

    # vertical lines at an interval of "line_distance" pixel
    for x in range(0,canvasWidth):
        canvas.create_line(x*line_distanceX, 0, x*line_distanceX, canvasHeight, fill="gray")
    # horizontal lines at an interval of "line_distance" pixel
    for y in range(0,canvasHeight):
        canvas.create_line(0, y*line_distanceY, canvasWidth, y*line_distanceY, fill="gray")


# Functions that calculate the master impedance array for given geometry
def n_getImpedanceArray(distanceArray, freq, dXmm, dYmm, lenght=1000, temperature=20, sigma20C=58e6, temCoRe=3.9e-3):
    '''
    Calculate the array of impedance as complex values for each element
    Input:
    distanceArray -  array of distances beetween the elements in [mm]
    freq = frequency in Hz
    dXmm - size of element in x [mm]
    dYmm - size of element in Y [mm]
    lenght - analyzed lenght in [mm] /default= 1000mm
    temperature - temperature of the conductors in deg C / defoult = 20degC
    sigma20C - conductivity of conductor material in 20degC in [S] / default = 58MS (copper)
    temCoRe - temperature resistance coeficcient / default is copper
    '''
    omega = 2*np.pi*freq

    impedanceArray = np.zeros((distanceArray.shape),dtype=np.complex_)
    for X in range(distanceArray.shape[0]):
        for Y in range(distanceArray.shape[0]):
            if X == Y:
                impedanceArray[Y, X] = n_getResistance(sizeX=dXmm, sizeY=dYmm, lenght=lenght, temp=temperature, sigma20C=sigma20C, temCoRe=temCoRe) + 1j*omega*n_getSelfInductance(sizeX=dXmm, sizeY=dYmm, lenght=lenght)
            else:
                impedanceArray[Y, X] = 1j*omega*n_getMutualInductance(sizeX=dXmm, sizeY=dYmm, lenght=lenght, distance=distanceArray[Y,X])

    return impedanceArray

# Function for calculating resistance array
def n_getResistanceArray(elementsVector, dXmm, dYmm, lenght=1000, temperature=20,sigma20C=58e6, temCoRe=3.9e-3):
    '''
    Calculate the array of resistance values for each element
    Input:
    elementsVector - The elements vector as delivered by arrayVectorize
    dXmm - size of element in x [mm]
    dYmm - size of element in Y [mm]
    lenght - analyzed lenght in [mm] /default= 1000mm
    temperature - temperature of the conductors in deg C / defoult = 20degC
    sigma20C - conductivity of conductor material in 20degC in [S] / default = 58MS (copper)
    temCoRe - temperature resistance coeficcient / default is copper
    '''

    resistanceArray = np.zeros(elementsVector.shape[0])
    for element in range(elementsVector.shape[0]):
        resistanceArray[element] = n_getResistance(sizeX=dXmm, sizeY=dYmm, lenght=lenght, temp=temperature, sigma20C=sigma20C, temCoRe=temCoRe)
    return resistanceArray

# Function that increase the resolution of the main geometry array
def n_arraySlicer(inputArray, subDivisions=2):
    '''
    This function increase the resolution of the cross section array
    inputArray -  oryginal geometry matrix
    subDivisions -  number of subdivisions / factor of increase of resoluttion / default = 2
    '''
    return inputArray.repeat(subDivisions,axis=0).repeat(subDivisions,axis=1)

# Functions that calculate module of complex number
def n_getComplexModule(x):
    '''
    returns the module of complex number
    input: x - complex number
    if not a complex number is given as parameter then it return the x diretly

    '''
    if isinstance(x, complex):
        return np.sqrt(x.real**2 + x.imag**2)
    else:
        return x


# Procedure that plot the array to canvas
def n_printTheArray(dataArray, canvas):
    '''
    This procedure allows to print the array back to the graphical board
    usefull for redraw or draw loaded data
    Inputs:
    dataArray -  the array to display on canvas
    canvas - tkinter canvas object
    '''

    # Let's check the size
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    # Now we calculate the propper dX and dY for this array
    canvasHeight = canvas.winfo_height()
    canvasWidth  = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY

    # Now we cleanUp the field
    n_checkered(canvas, elementsInX, elementsInY)

    for Row in range(elementsInY):
        for Col in range(elementsInX):
            if dataArray[Row][Col] == 1:
                fillColor = "red"
                canvas.create_rectangle((Col)*dX, (Row)*dY, (Col)*dX+dX, (Row)*dY+dY, fill=fillColor, outline="gray")

            elif dataArray[Row][Col] == 2:
                fillColor = "green"
                canvas.create_rectangle((Col)*dX, (Row)*dY, (Col)*dX+dX, (Row)*dY+dY, fill=fillColor, outline="gray")

            elif dataArray[Row][Col] == 3:
                fillColor = "blue"
                canvas.create_rectangle((Col)*dX, (Row)*dY, (Col)*dX+dX, (Row)*dY+dY, fill=fillColor, outline="gray")

# Procedure to set up point in the array and display it on canvas
def n_setUpPoint(event, Set, dataArray, canvas):
    '''
    This procedure track the mouse position from event ad setup or reset propper element
    in the cross section array
    Inputs
    event - the event object from tkinter that create the point (or reset)
    Set - Number of phase to set or 0 to reset
    dataArray -  the array that keeps the cross section design data
    canvas - tk inter canvas object
    '''
    # gathering some current data
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY

    Col = int(event.x/dX)
    Row = int(event.y/dY)

    if event.x < canvasWidth and event.y < canvasHeight and event.x > 0 and event.y > 0:
        inCanvas = True
    else:
        inCanvas = False

    if Set != 0 and inCanvas:
        actualPhase = Set

        if actualPhase == 3:
            canvas.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="blue", outline="gray")
            dataArray[Row][Col] = 3
        elif actualPhase == 2:
            canvas.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="green", outline="gray")
            dataArray[Row][Col] = 2
        else:
            canvas.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="red", outline="gray")
            dataArray[Row][Col] = 1

    elif Set==0 and inCanvas:
        canvas.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="white", outline="gray")
        dataArray[Row][Col] = 0

# Function that put back together the solution vectr back to represent the crss section shape array
def n_recreateresultsArray(elementsVector, resultsVector, initialGeometryArray):
    '''
    Functions returns recreate cross section array with mapperd solution results
    Inputs:
    elementsVector - vector of crossection elements as created by the n_arrayVectorize
    resultsVector - vectr with results values calculated base on the elementsVector
    initialGeometryArray - the array that contains the cross section geometry model
    '''
    localResultsArray = np.zeros((initialGeometryArray.shape), dtype=float)

    for vectorIndex, result in enumerate(resultsVector):
        localResultsArray[int(elementsVector[vectorIndex][0]),int(elementsVector[vectorIndex][1])] = result

    return localResultsArray

def n_sumVecList(list):
        sumV = v2(0, 0)
        for v in list:
            sumV = sumV + v
        return sumV

def n_getForces(XsecArr, vPhA, vPhB, vPhC, Ia, Ib, Ic, Lenght=1):
    '''
    this experimental functions will calcuate the fore vector for each phase
    in given geometry and currents values.
    Inputs:
    vPhA/B/C - elements vectors of the each phase geometry as delivered by n_arrayVectorize
    Ia/b/c - current value in each phase in [A]
    '''
    def sumVecList(list):
        sumV = v2(0, 0)
        for v in list:
            sumV = sumV + v
        return sumV

    mi0_o2pi = 2e-7
    lPh = (len(vPhA), len(vPhB), len(vPhC))  # Memorizing each phaze elements count
    Iph = (Ia / len(vPhA), Ib / len(vPhB), Ic / len(vPhC))

    vPhAll = np.concatenate((vPhA, vPhB, vPhC), axis=0)  # One vector for all phases

    totalForceVec = []



    for this in vPhAll:
        forceVec = v2(0, 0)  # initial reset for this element force
        for other in vPhAll:
            if this[0] != other[0] or this[1] != other[1]:
                distV = v2(other[2]-this[2], other[3]-this[3])
                direction = distV.normalize()
                distance = distV.norm() * 1e-3  # to convert into [m]

                Ithis = Iph[int(XsecArr[int(this[0])][int(this[1])])-1]
                Iother = Iph[int(XsecArr[int(other[0])][int(other[1])])-1]

                forceVec += Lenght * (mi0_o2pi * Iother * Ithis / distance) * direction

        totalForceVec.append(forceVec)

    ForceA = sumVecList(totalForceVec[:lPh[0]])
    ForceB = sumVecList(totalForceVec[lPh[0]: lPh[0] + lPh[1]])
    ForceC = sumVecList(totalForceVec[lPh[0] + lPh[1]:])

    ForceMagVect = [force.norm() for force in totalForceVec]

    return ForceA, ForceB, ForceC, ForceMagVect, totalForceVec

def n_getPhasesCenters(vPhA, vPhB, vPhC):
    '''
    This functions calculate the geometry center (average) for each phase
    delivered as a vector form
    Inputs:
    vPhA/B/C - elements vectors of the each phase geometry as delivered by n_arrayVectorize
    '''
    tempX = [x[2] for x in vPhA]
    tempY = [x[3] for x in vPhA]
    Pha = (sum(tempX) / len(tempX), sum(tempY) / len(tempY))

    tempX = [x[2] for x in vPhB]
    tempY = [x[3] for x in vPhB]
    Phb = (sum(tempX) / len(tempX), sum(tempY) / len(tempY))

    tempX = [x[2] for x in vPhC]
    tempY = [x[3] for x in vPhC]
    Phc = (sum(tempX) / len(tempX), sum(tempY) / len(tempY))

    return Pha, Phb, Phc

def n_getCenter(v):
    '''
    This functions calculate the geometry center (average) for each phase
    delivered as a vector form
    Inputs:
    vPhA/B/C - elements vectors of the each phase geometry as delivered by n_arrayVectorize
    '''
    tempX = [x[2] for x in v]
    tempY = [x[3] for x in v]
    center = (sum(tempX) / len(tempX), sum(tempY) / len(tempY))

    return center


def n_getConductors(XsecArr, vPhA, vPhB, vPhC):
    '''
    [Row,Col,X,Y]
    '''
    # Setting up new conductors array
    conductorsArr = np.zeros((XsecArr.shape), dtype=int)

    conductor = 0
    phases = [vPhA, vPhB, vPhC]
    phaseCond = []

    for phase in phases:
        phaseConductors = 0

        for element in phase:
            R = int(element[0])
            C = int(element[1])

            if conductorsArr[R, C] == 0:
                # tests in 4 directions
                N, E, S, W = 0, 0, 0, 0
                try:
                    E = conductorsArr[R+1, C]
                    W = conductorsArr[R-1, C]
                    N = conductorsArr[R, C-1]
                    S = conductorsArr[R, C+1]
                except:
                    pass
                if N != 0:
                    conductorsArr[R, C] = N
                elif S != 0:
                    conductorsArr[R, C] = S
                elif E != 0:
                    conductorsArr[R, C] = E
                elif W != 0:
                    conductorsArr[R, C] = W
                else:
                    conductor += 1
                    phaseConductors += 1
                    conductorsArr[R, C] = conductor
        phaseCond.append(phaseConductors)

    return conductorsArr, conductor, phaseCond
