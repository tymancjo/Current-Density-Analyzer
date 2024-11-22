"""
This is the functions and procedure library file to be used in the CDS solution
It is intended to simplify the way the code is written and making it 
easy to follow and understand.

Author: Tomasz Tomanek | 2024
"""

import os
import pickle
import numpy as np
from csdlib import innercode as ic


def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, *args, *kwargs)

def getCanvas(textInput):
    """This function is to determine the best parameters for the canvas
    based on the given geometry steps defined by the inner code."""

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

        # defining the array size and scale params
        dXmm = dYmm = xd
        XSecArray = np.zeros([elements_x, elements_y])

        # adding the defiend cells to the geometry array
        for step in codeSteps:
            step[0](*step[1], shift=(min(X), min(Y)), XSecArray=XSecArray, dXmm=dXmm)
        return XSecArray, dXmm, dYmm

    return False


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



def combineVectors(vPhA, vPhB, vPhC):
    """Function is joining the 3 phase vectors together"""

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
    
    return elementsVector, elementsPhaseA,elementsPhaseB, elementsPhaseC


