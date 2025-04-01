"""
This is the functions and procedure library file to be used in the CDS solution
It is intended to simplify the way the code is written and making it 
easy to follow and understand.

Author: Tomasz Tomanek | 2024
"""

import os
import sys
import pickle
import numpy as np
from csdlib import innercode as ic


class cointainer:
    """
    This class is to make an object to operate with the loading and saving the
    .csd files - that have the geometry data
    """

    def __init__(self, xArray, dX, dY, ic=""):
        self.xArray = xArray
        self.dX = dX
        self.dY = dY
        self.ic = ic
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

    def getIC(self):
        return self.ic

class the_bar:
    def __init__(self):
        self.elements = []
        self.center = []
        self.power = 0
        self.current = 0
        self.number = 0
        self.phase = 0
        self.perymiter = 0

def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, *args, *kwargs)

def getCanvas(codeSteps):
    """This function is to determine the best parameters for the canvas
    based on the given geometry steps defined by the inner code."""

    # codeLines = textInput.splitlines()
    # codeSteps, currents = ic.textToCode(codeLines)

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
            sizes = [ 10, 5, 4, 2.5, 2, 1]
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

    currents = []
    materials = []
    if os.path.isfile(filename):
        _, extension = os.path.splitext(filename)
        myLog("File type: " + extension)

        if extension.lower() in [".txt", ".inc", ".ic"]:
            myLog("reading the inner-code geometry file: " + filename)
            try:
                with open(filename, "r") as f:
                    file_content = f.read()

                codeLines = file_content.splitlines()
                codeSteps, currents, materials = ic.textToCode(codeLines)

                XSecArray, dXmm, dYmm = getCanvas(codeSteps)
            except IOError:
                print("Error reading the file " + filename)
                sys.exit(1)

        else:
            myLog("reading from file :" + filename)
            XSecArray, dXmm, dYmm = loadObj(filename).restore()

        return XSecArray, dXmm, dYmm, currents, materials
    else:
        print(f"The file {filename} can't be opened!")
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

def combineVectors(*list_of_vectors):
    # previous version was unnecessary over-complicated 
    # It stays as a function for the compatibility with the code reasons.

    lengths = [len(v) for v in list_of_vectors]
    output = np.concatenate(list_of_vectors)

    return output, *lengths,

def getConductors(XsecArr, phases):
    """
    [Row,Col,X,Y]
    """
    # Setting up new conductors array
    conductorsArr = np.zeros((XsecArr.shape), dtype=int)

    conductors_number = 0
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
                                    N =  conductorsArr[max(0,R+dR),max(0,altered_C+dC)]
                                except:
                                    N = 0
                                if N > 0:
                                    conductorsArr[R,C] = N
                                    break
                        elif conductorsArr[R,altered_C] == 0:
                            break


                    if conductorsArr[R,C] < 1:
                        # if we didn't find any neighbor already marked. 
                        conductors_number += 1
                        phaseConductors += 1
                        this_phase_cond_numbers.append(conductors_number)
                        conductorsArr[R, C] = conductors_number


        phaseCond.append(this_phase_cond_numbers)


    return conductorsArr, conductors_number, phaseCond

def getPerymiter(vec, arr, dXmm, dYmm):

    """
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
    """
    # TODO: adding check if we dont exeed dimensions of array
    # its done
    perymiter = 0
    for box in vec:
        # checking the size of the arr array
        x, y = arr.shape

        # checking in x directions lef and right
        A, B = int(box[0] + 1), int(box[1])

        if A < x:
            if arr[A][B] == 0:
                perymiter += dYmm
        else:
            perymiter += dYmm

        A, B = int(box[0] - 1), int(box[1])
        if A >= 0:
            if arr[A][B] == 0:
                perymiter += dYmm
        else:
            perymiter += dYmm

        A, B = int(box[0]), int(box[1] + 1)
        if B < y:
            if arr[A][B] == 0:
                perymiter += dXmm
        else:
            perymiter += dXmm

        A, B = int(box[0]), int(box[1] - 1)

        if B >= 0:
            if arr[A][B] == 0:
                perymiter += dXmm
        else:
            perymiter += dXmm

    return perymiter

def plot_the_geometry(DataArray, ax,cmap,  dXmm=10, dYmm=10):

    num_ticks_x = len(DataArray[0])
    num_ticks_y = len(DataArray)

    # Set the ticks and corresponding labels
    step = int(num_ticks_x / (num_ticks_x * dXmm / 10))

    x_ticks = np.arange(0, num_ticks_x, step)
    y_ticks = np.arange(0, num_ticks_y, step)

    # Set the ticks based on the array dimensions
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set the tick labels by multiplying
    # the tick values by the scaling factor
    ax.set_xticklabels((x_ticks * dXmm).astype(int))
    ax.set_yticklabels((y_ticks * dYmm).astype(int))

    # plt.imshow(XSecArray, cmap=cmap, norm=norm)
    return ax.imshow(DataArray, cmap=cmap)





