"""
This is separate library file for the CSD.py applications
Its done this way to clean up the code in the main app

v00 - Initial Build
"""

# External Loads
from csdlib.vect import Vector as v2
import os
import numpy as np
import functools
from tkinter import filedialog, messagebox
from tkinter import *
import pickle
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("TKAgg")


# End of external Loads
from csdlib import csdmath as csdm

# Redirecting to optimized csdmath versions
def n_getMutualInductance(sizeX, sizeY, lenght, distance):
    return csdm.getMutualInductance(sizeX, sizeY, lenght, distance)

def n_getSelfInductance(sizeX, sizeY, lenght):
    return csdm.getSelfInductance(sizeX, sizeY, lenght)

def n_getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    return csdm.getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe)

def n_getDistancesArray(inputVector):
    return csdm.getDistancesArray(inputVector)

def n_getImpedanceArray(distanceArray, freq, dXmm, dYmm, lenght=1000, temperature=20, sigma20C=58e6, temCoRe=3.9e-3):
    return csdm.getImpedanceArray(distanceArray, freq, dXmm, dYmm, lenght, temperature, sigma20C, temCoRe)

def n_getResistanceArray(elementsVector, dXmm, dYmm, lenght=1000, temperature=20, sigma20C=58e6, temCoRe=3.9e-3):
    return csdm.getResistanceArray(elementsVector, dXmm, dYmm, lenght, temperature, sigma20C, temCoRe)

def n_arrayVectorize(inputArray, phaseNumber, dXmm, dYmm):
    return csdm.arrayVectorize(inputArray, phaseNumber, dXmm, dYmm)

def n_recreateresultsArray(elementsVector, resultsVector, initialGeometryArray):
    return csdm.recreateresultsArray(elementsVector, resultsVector, initialGeometryArray)

def n_getComplexModule(x):
    return csdm.getComplexModule(x)

def n_arraySlicer(inputArray, subDivisions=2):
    return csdm.arraySlicer(inputArray, subDivisions)

def n_perymiter(vec, arr, dXmm, dYmm):
    if len(vec) > 0:
        row, col = int(vec[0, 0]), int(vec[0, 1])
        phase_id = arr[row, col]
        return csdm.getPerymiter(arr, dXmm, dYmm, phase_id=phase_id)
    return 0

# Classes


class cointainer:
    def __init__(self, xArray, dX, dY):
        self.xArray = xArray
        self.dX = dX
        self.dY = dY
        print("The container is created")

    def save(self, filename):
        if filename[-4:] == ".csd":  # the name already have extension
            pass
        else:
            filename = str(filename) + ".csd"
        try:
            print("trying to save to: {}".format(filename))
            tempfile = filename + "_temp"
            with open(tempfile, "wb") as output:
                pickle.dump(self, output, pickle.DEFAULT_PROTOCOL)
        except:
            print("There is issue with pickle. Save aborted to protect file.")
        else:
            with open(filename, "wb") as output:
                pickle.dump(self, output, pickle.DEFAULT_PROTOCOL)
            ## If file exists, delete it ##
            if os.path.isfile(tempfile):
                os.remove(tempfile)
            else:  ## Show an error ##
                print("Error: %s file not found" % tempfile)

    def restore(self):
        return self.xArray, self.dX, self.dY


# ################# FUNCTIONS & PROCEDURES##############################


def n_shiftPhase(phaseId, dX, dY, XSecArray,remain=0):
    """
    This procedure is shifting the particucal geometry of the phase in arrays
    to the specific x and y direction.
    input:
    phaseId - the value in the geometry array t hat describes the phase 1,2 or 3
    dX - number of cells to shift in columnspan
    dY - number of cells to shift in rows
    XSecArray - input geometry array
    """

    # making the copy of input geommetry array
    tempGeometry = np.copy(XSecArray)
    # deleting the other phases geometry from the array
    tempGeometry[tempGeometry != phaseId] = 0
    # deleting the selected phase in original geometry array
    XSecArray[XSecArray == phaseId] = remain

    oR = XSecArray.shape[0]
    oC = XSecArray.shape[1]

    for r in range(oR):
        for c in range(oC):
            if tempGeometry[r, c] == phaseId:
                nR = r + dY
                nC = c + dX
                if nR >= oR:
                    nR -= oR
                if nC >= oC:
                    nC -= oC
                XSecArray[nR, nC] = tempGeometry[r, c]


def n_cloneGeometry(dX, dY, N, XSecArray):
    """
    This procedure alternate the x section array multiplying the
    existing geometry as a pattern with defined shift vector
    in cells
    input:
    dX - shift of cells in X (cols)
    dy - shift of cells in Y (rows)
    N - number of copies created
    XSecArray - input array of the base cross section
    """
    # Lets figure out the new shape of the array
    # Original shape
    oR = XSecArray.shape[0]
    oC = XSecArray.shape[1]

    d = max(dX, dY)

    # the new shape can be figured out by the relation
    nR = N * d + oR
    nC = N * d + oC

    print("New array shape: {}x{}".format(nR, nC))

    # creating new empty array of required size
    NewGeometryArray = np.zeros((nR, nC))

    # placing the existing array into the new one as copies
    NewGeometryArray[0:oR, 0:oC] = XSecArray
    for x in range(1, N + 1):
        print("copying to: {} x {}".format(x * dY, x * dX))
        for row in range(x * dY, x * dY + oR):
            for col in range(x * dX, x * dX + oC):
                if XSecArray[row - x * dY, col - x * dX] != 0:
                    NewGeometryArray[row, col] = XSecArray[row - x * dY, col - x * dX]

    # and sign new array back to the main one of geometry
    return NewGeometryArray


def n_getDistance(A, B):
    """
    This function returns simple in line distance between two points
    defined by input
    input:
    A - tuple (x,y) - first point position on surface
    B - tuple (x,y) - first point position on surface

    Output
    Distance between A and B in the units of position
    """

    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


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


# Calculations of mututal inductance between conductors
# Calculation of self inductance value function
# Calculate the resistance value function
# Calculate distance between elements function
# Master Array Vectorization FUNCTION
# Functions that calculate the master impedance array for given geometry
# Function for calculating resistance array
# Function that increase the resolution of the main geometry array
# Functions that calculate module of complex number


def n_checkered(canvas, cutsX, cutsY, mode=0):
    """
    This function clean the board and draw grid
    Inputs:
    canvas - tkinter canvas object
    cutsX - elements in X (left right) direction
    cutsY - elements in Y (top down) direction
    """

    # Reading the size of the canvas element
    canvasHeight = canvas.winfo_height()
    canvasWidth = canvas.winfo_width()

    line_distanceX = canvasWidth / cutsX
    line_distanceY = canvasHeight / cutsY

    # Cleaning up the whole canvas space by drawing a white rectangle
    if mode == 0 or mode == 1:
        canvas.create_rectangle(
            0, 0, canvasWidth, canvasHeight, fill="white", outline="gray"
        )

    # vertical lines at an interval of "line_distance" pixel
    # some limits added - we dont draw it if the line amout is to big
    # it would be mess anyway if too much
    if max(cutsX, cutsY) <= 100 and mode == 0 or max(cutsX, cutsY) <= 100 and mode == 2:
        for x in range(0, cutsX):
            canvas.create_line(
                x * line_distanceX, 0, x * line_distanceX, canvasHeight, fill="gray"
            )
        # horizontal lines at an interval of "line_distance" pixel
        for y in range(0, cutsY):
            canvas.create_line(
                0, y * line_distanceY, canvasWidth, y * line_distanceY, fill="gray"
            )


# Procedure to set up point in the array and display it on canvas
def n_setUpPoint(event, Set, dataArray, canvas):
    """
    This procedure track the mouse position from event ad setup or reset propper element
    in the cross section array
    Inputs
    event - the event object from tkinter that create the point (or reset)
    Set - Number of phase to set or 0 to reset
    dataArray -  the array that keeps the cross section design data
    canvas - tk inter canvas object
    """
    # gathering some current data
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY

    Col = int(event.x / dX)
    Row = int(event.y / dY)

    if event.x < canvasWidth and event.y < canvasHeight and event.x > 0 and event.y > 0:
        inCanvas = True
    else:
        inCanvas = False

    if Set != 0 and inCanvas:
        actualPhase = Set

        if actualPhase == 3:
            canvas.create_rectangle(
                Col * dX,
                Row * dY,
                Col * dX + dX,
                Row * dY + dY,
                fill="blue",
                outline="gray",
            )
            dataArray[Row][Col] = 3
        elif actualPhase == 2:
            canvas.create_rectangle(
                Col * dX,
                Row * dY,
                Col * dX + dX,
                Row * dY + dY,
                fill="green",
                outline="gray",
            )
            dataArray[Row][Col] = 2
        else:
            canvas.create_rectangle(
                Col * dX,
                Row * dY,
                Col * dX + dX,
                Row * dY + dY,
                fill="red",
                outline="gray",
            )
            dataArray[Row][Col] = 1

    elif Set == 0 and inCanvas:
        canvas.create_rectangle(
            Col * dX,
            Row * dY,
            Col * dX + dX,
            Row * dY + dY,
            fill="white",
            outline="gray",
        )
        dataArray[Row][Col] = 0


# Function that put back together the solution vectr back to represent the crss section shape array
def n_sumVecList(list):
    sumV = v2(0, 0)
    for v in list:
        sumV = sumV + v
    return sumV


def n_getForces(XsecArr, vPhA, vPhB, vPhC, Ia, Ib, Ic, Lenght=1):
    """
    this experimental functions will calcuate the fore vector for each phase
    in given geometry and currents values.
    Inputs:
    vPhA/B/C - elements vectors of the each phase geometry as delivered by n_arrayVectorize
    Ia/b/c - current value in each phase in [A]
    """

    def sumVecList(list):
        sumV = v2(0, 0)
        for v in list:
            sumV = sumV + v
        return sumV

    mi0_o2pi = 2e-7
    # Memorizing each phaze elements count
    lPh = (len(vPhA), len(vPhB), len(vPhC))
    Iph = (Ia / len(vPhA), Ib / len(vPhB), Ic / len(vPhC))

    # One vector for all phases
    vPhAll = np.concatenate((vPhA, vPhB, vPhC), axis=0)

    totalForceVec = []

    for this in vPhAll:
        forceVec = v2(0, 0)  # initial reset for this element force
        for other in vPhAll:
            if this[0] != other[0] or this[1] != other[1]:
                distV = v2(other[2] - this[2], other[3] - this[3])
                direction = distV.normalize()
                distance = distV.norm() * 1e-3  # to convert into [m]

                Ithis = Iph[int(XsecArr[int(this[0])][int(this[1])]) - 1]
                Iother = Iph[int(XsecArr[int(other[0])][int(other[1])]) - 1]

                forceVec += Lenght * (mi0_o2pi * Iother * Ithis / distance) * direction

        totalForceVec.append(forceVec)

    ForceA = sumVecList(totalForceVec[: lPh[0]])
    ForceB = sumVecList(totalForceVec[lPh[0] : lPh[0] + lPh[1]])
    ForceC = sumVecList(totalForceVec[lPh[0] + lPh[1] :])

    ForceMagVect = [force.norm() for force in totalForceVec]

    return ForceA, ForceB, ForceC, ForceMagVect, totalForceVec


def n_getPhasesCenters(vPhA, vPhB, vPhC):
    """
    This functions calculate the geometry center (average) for each phase
    delivered as a vector form
    Inputs:
    vPhA/B/C - elements vectors of the each phase geometry as delivered by n_arrayVectorize
    """
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
    """
    This functions calculate the geometry center (average) for each phase
    delivered as a vector form
    Inputs:
    vPhA/B/C - elements vectors of the each phase geometry as delivered by n_arrayVectorize
    """
    tempX = [x[2] for x in v]
    tempY = [x[3] for x in v]
    center = (sum(tempX) / len(tempX), sum(tempY) / len(tempY))

    return center


def n_getConductors_old(XsecArr, vPhA, vPhB, vPhC):
    """
    [Row,Col,X,Y]
    """
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
                    E = conductorsArr[R + 1, C]
                    W = conductorsArr[R - 1, C]
                    N = conductorsArr[R, C - 1]
                    S = conductorsArr[R, C + 1]
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

def n_getConductors(XsecArr, vPhA, vPhB, vPhC):
    """
    [Row,Col,X,Y]
    """
    # Setting up new conductors array
    conductorsArr = np.zeros((XsecArr.shape), dtype=int)

    conductor = 0
    phases = [vPhA, vPhB, vPhC]
    phaseCond = []

    # let's map the elements back to the 2D shape array.
    # I can use the recreate array here - but for sake of the clarity
    phase_numbers = []
    for phase_number, phase in enumerate(phases):

        phase_numbers.append(-1-phase_number)

        for element in phase:
            R = int(element[0])
            C = int(element[1])

            conductorsArr[R, C] = phase_numbers[-1]
    
    Rows, Cols = conductorsArr.shape

    # look around coordinates vector
    dRC = [(-1,-1), (-1,0),(-1,1),
            (0,-1),  (0,1),
            (1,-1), (1,0),(1,1)] 

    for phase_number in phase_numbers:
        phaseConductors = 0
        this_phase_cond_numbers = []
        for R in range(Rows):
            for C in range(Cols):
                if conductorsArr[R,C] == phase_number:
                    # looking around
                    for step in range(5):
                        altered_C = min(C+step, Cols-1)
                        # to look around more if we follow a conductor of phases
                        if conductorsArr[R,altered_C] == phase_number or conductorsArr[R,altered_C] in this_phase_cond_numbers :
                            for dR,dC in dRC:
                                # just to be able not fall of the size of array
                                try:
                                    N =  conductorsArr[R+dR,altered_C+dC]
                                except:
                                    N = 0
                                if N > 0:
                                    conductorsArr[R,C] = N
                                    break
                    if conductorsArr[R,C] < 1:
                        # if we didn't find any neighbor already marked. 
                        conductor += 1
                        phaseConductors += 1
                        this_phase_cond_numbers.append(conductor)
                        conductorsArr[R, C] = conductor


        phaseCond.append(phaseConductors)


    return conductorsArr, conductor, phaseCond