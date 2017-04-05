'''
This is separate library file for the CSD.py applications
Its done this way to clean up the code in the main app

v00 - Initial Build
'''

### External Loads
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog, messagebox

import functools
import numpy as np

import os.path

### End of external Loads

# ################# FUNCTIONS & PROCEDURES##############################


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
    return (lenght/(sizeX*sizeY*sigma20C)) * 1e3 *(1+temCoRe*(temp-20))

# Calculate distance between elements function
def n_getDistancesArray(inputVector):
    '''
    This function calculate the array of distances between every conductors element
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
                posXa =  inputVector[y][2]
                posYa =  inputVector[y][3]

                posXb =  inputVector[x][2]
                posYb =  inputVector[x][3]

                distanceArray[y,x] = np.sqrt((posXa-posXb)**2 + (posYa-posYb)**2)
            else:
                distanceArray[y,x] = 0
    return distanceArray


# Master Array Vecrorization FUNCTION
def n_arrayVectorize(inputArray,phaseNumber, dXmm, dYmm):
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

                vectorArray.append([Row,Col,coordinateX,coordinateY])

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

    line_distanceX = int(canvasWidth / cutsX)
    line_distanceY = int(canvasHeight / cutsY)


    # Cleaning up the whole canvas space by drawing a white rectangle
    canvas.create_rectangle(0, 0, canvasWidth, canvasHeight, fill="white", outline="gray")

    # vertical lines at an interval of "line_distance" pixel
    for x in range(0,canvasWidth,line_distanceX):
        canvas.create_line(x, 0, x, canvasHeight, fill="gray")
    # horizontal lines at an interval of "line_distance" pixel
    for y in range(0,canvasHeight,line_distanceY):
        canvas.create_line(0, y, canvasWidth, y, fill="gray")


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
