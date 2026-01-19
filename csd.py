import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
import os.path
import time

# Importing local library
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2
from csdlib import csdgui as gui
from csdlib import csdcli as cli
from csdlib import innercode as ic


def showXsecArray(event):
    """
    This function print the array to the terminal
    """
    print(XSecArray)


def saveArrayToFile():
    """
    This function saves the data of cross section array to file
    """
    filename = filedialog.asksaveasfilename()
    filename = os.path.normpath(filename)
    if filename:
        saveTheData(filename)


def saveTheData(filename):
    """
    This is the subfunction for saving data
    """
    # print('Saving to file :' + filename)
    # np.save(filename, XSecArray)

    S = csd.cointainer(XSecArray, dXmm, dYmm)
    S.save(filename)
    del S


def loadArrayFromFile():
    """
    This function loads the data from the file
    !!!!! Need some work - it dosn't reset properly the dXmm and dYmm
    """
    filename = filedialog.askopenfilename(filetypes=[("CSD files", "*.csd")])
    filename = os.path.normpath(filename)

    if (
        os.path.isfile(filename) and np.sum(XSecArray) != 0
    ):  # Test if there is anything draw on the array
        q = messagebox.askquestion(
            "Delete", "This will delete current shape. Are You Sure?", icon="warning"
        )
        if q == "yes":
            loadTheData(filename)
    else:
        if os.path.isfile(filename):
            loadTheData(filename)


def importArrayFromPicture():
    """
    This function triggers the importing data from a picture process
    """
    filename = filedialog.askopenfilename(
        filetypes=[("Pictures", "*.jpg *.jpeg *.JPG *.PNG *.png")]
    )
    filename = os.path.normpath(filename)

    if (
        os.path.isfile(filename) and np.sum(XSecArray) != 0
    ):  # Test if there is anything draw on the array
        q = messagebox.askquestion(
            "Delete", "This will delete current shape. Are You Sure?", icon="warning"
        )
        if q == "yes":
            importTheData(filename)
    else:
        if os.path.isfile(filename):
            importTheData(filename)


def loadTheData(filename):
    """
    This is sub function to load data
    """
    global XSecArray, dXmm, dYmm

    print("reading from file :" + filename)
    # XSecArray =  np.load(filename)
    S = csd.loadObj(filename)
    XSecArray, dXmm, dYmm = S.restore()
    print("dX:{} dY:{}".format(dXmm, dYmm))
    myEntryDx.delete(0, END)
    myEntryDx.insert(END, str(dXmm))
    setParameters()
    printTheArray(XSecArray, canvas=w)
    del S


def importTheData(filename):
    """
    This function is to import the picture as a geometry data
    """
    global XSecArray, dXmm, dYmm

    img_w = img_h = 1000
    config = {"image": filename, "usepil": False, "show": True}
    array_img = cli.loadImageFromFile(config)

    pixels_x = array_img.shape[0]
    pixels_y = array_img.shape[1]

    dXmm = dYmm = min(img_w / pixels_x, img_h / pixels_y)

    print(f"Imported image: {filename}")
    print(f"Image size: {pixels_x} by {pixels_y} pixels")
    print(f"Cell size dx: {dXmm}mm dy: {dYmm}mm")

    XSecArray = cli.getCSD(array_img)

    if np.sum(XSecArray) == 0:
        print("No cross section data found. No output generated.")
    else:
        XSecArray = cli.trimEmpty(XSecArray)
        XSecArray, dXmm, dYmm, _ = cli.simplify(XSecArray, dXmm, dYmm, maxsize=500)
        XSecArray = cli.trimEmpty(XSecArray)

        myEntryDx.delete(0, END)
        myEntryDx.insert(END, str(dXmm))

        setParameters()
        printTheArray(XSecArray, canvas=w)


def zoomInArray(inputArray, zoomSize=2, startX=0, startY=0):
    oryginalX = inputArray.shape[0]
    oryginalY = inputArray.shape[1]

    NewX = oryginalX // zoomSize
    NewY = oryginalY // zoomSize

    if startX > (oryginalX - NewX):
        startX = oryginalX - NewX

    if startY > (oryginalY - NewY):
        startY = oryginalY - NewY

    return inputArray[startY : startY + NewY, startX : startX + NewX]


def zoomIn():
    global globalZoom

    if globalZoom < 5:
        globalZoom += 1

    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def zoomOut():
    global globalZoom, globalX, globalY

    if globalZoom > 1:
        globalZoom -= 1

    globalX -= 2
    if globalX < 0:
        globalX = 0

    globalY -= 2
    if globalY < 0:
        globalY = 0

    if globalZoom == 1:
        globalX = 0
        globalY = 0

    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def redraw(*args):
    # global globalX, globalY
    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def zoomL():
    global globalX, globalY

    globalX -= 2
    if globalX < 0:
        globalX = 0
    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def zoomR():
    global globalX, globalY

    globalX += 2

    if globalX > XSecArray.shape[1] - XSecArray.shape[1] // globalZoom:
        globalX = XSecArray.shape[1] - XSecArray.shape[1] // globalZoom

    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def zoomU():
    global globalX, globalY

    globalY -= 2
    if globalY < 0:
        globalY = 0

    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def zoomD():
    global globalX, globalY

    globalY += 2
    if globalY > XSecArray.shape[0] - XSecArray.shape[0] // globalZoom:
        globalY = XSecArray.shape[0] - XSecArray.shape[0] // globalZoom

    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)


def displayArrayAsImage():
    """
    This function print the array to termianl and shows additional info of the
    dX and dy size in mm
    and redraw the array on the graphical working area
    """
    print(XSecArray)
    print(str(dXmm) + "[mm] :" + str(dYmm) + "[mm]")
    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)

    drawGeometryArray(XSecArray)


def clearArrayAndDisplay():
    """
    This function erase the datat form array and return it back to initial
    setup
    """
    global XSecArray, dX, dY
    if np.sum(XSecArray) != 0:  # Test if there is anything draw on the array
        q = messagebox.askquestion(
            "Delete", "This will delete current shape. Are You Sure?", icon="warning"
        )
        if q == "yes":
            XSecArray = np.zeros(XSecArray.shape)
            # checkered(w, dX, dY)
            mainSetup()
            # csd.n_checkered(w, elementsInX, elementsInY)
            printTheArray(XSecArray, w)
            myEntryDx.delete(0, END)
            myEntryDx.insert(END, str(dXmm))
            setParameters()

    else:
        XSecArray = np.zeros(XSecArray.shape)
        # checkered(w, dX, dY)
        mainSetup()
        # csd.n_checkered(w, elementsInX, elementsInY)
        printTheArray(XSecArray, w)
        myEntryDx.delete(0, END)
        myEntryDx.insert(END, str(dXmm))
        setParameters()


def subdivideArray():
    """
    This function is logical wrapper for array slicer
    it take care to not loose any entered data from the modified array
    """
    global XSecArray, dXmm, dYmm, selectionArray

    if dXmm > 1 and dYmm > 1:
        XSecArray = csd.n_arraySlicer(inputArray=XSecArray, subDivisions=2)

        dXmm = dXmm / 2
        dYmm = dYmm / 2

        # just maiking the lock fo the smallest 1mm size.
        dXmm = max(dXmm, 1)
        dYmm = max(dYmm, 1)

        selectionArray = None

        print(str(dXmm) + "[mm] :" + str(dYmm) + "[mm]")
        printTheArray(dataArray=XSecArray, canvas=w)
    else:
        print("No further subdivisions make sense :)")

    myEntryDx.delete(0, END)
    myEntryDx.insert(END, str(dXmm))
    setParameters()

    # end= time.clock()
    # print('subdiv time :'+str(end - start))


def simplifyArray():
    """
    This function simplified array - but it take more care to not loose any data
    entered by user
    """
    global XSecArray, dXmm, dYmm, selectionArray

    if dXmm < 30 and dYmm < 30:
        XSecArray = XSecArray[::2, ::2]

        dXmm = dXmm * 2
        dYmm = dYmm * 2

        selectionArray = None
        # print(str(dXmm)+'[mm] :'+str(dYmm)+'[mm]')
        printTheArray(dataArray=XSecArray, canvas=w)
    else:
        print("No further simplification make sense :)")

    myEntryDx.delete(0, END)
    myEntryDx.insert(END, str(dXmm))
    setParameters()


def extendArray():
    """This function modifies the original array of geometry
    and extends it by adding a empty space around the existing range
    it is adding extension (10) rows/cols each side"""
    global XSecArray

    rows = XSecArray.shape[0]
    cols = XSecArray.shape[1]

    extension = 10

    extendArray = np.zeros((rows + 2 * extension, cols + 2 * extension))
    extendArray[extension : extension + rows, extension : extension + cols] = XSecArray
    XSecArray = extendArray
    zoomOut()


def showMeForces(*arg):
    """
    This function trigger the forceWindow object to make it possible for
    Icw forces calculations
    """
    # Maiking it global - as for now it's not all object based gui
    global forceCalc

    # Read the setup params from GUI
    setParameters()

    # lets check if there is anything in the xsection geom array
    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Forces calculator")
        forceCalc = gui.forceWindow(root, XSecArray, dXmm, dYmm)


def showMePower(*arg):
    # lets check if there is anything in the xsection geom array
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Power Losses Calculator")
        powerCalc = gui.currentDensityWindow(root, XSecArray, dXmm, dYmm)


def showMePro(*arg):
    # lets check if there is anything in the xsection geom array
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Pro Power Losses Solver")
        powerCalc = gui.currentDensityWindowPro(root, XSecArray, dXmm, dYmm)


def showMeZ(*arg):
    # lets check if there is anything in the xsection geom array
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Impednaces Calculator")
        zCalc = gui.zWindow(root, XSecArray, dXmm, dYmm)


def showMeZ3f(*arg):
    # lets check if there is anything in the xsection geom array
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Impednaces Calculator")
        zCalc = gui.zWindow3f(root, XSecArray, dXmm, dYmm)


def showReplacer(*arg):
    global XSecArray

    # TODO: New window phase switcher based on below
    root = Tk()
    root.title("Impednaces Calculator")
    TestWindow = gui.geometryModWindow(root, w)

    try:
        sourcePhase = int(input("Source phase [1,2,3]: "))
        toPhase = int(input("to phase [1,2,3]: "))
    except:
        sourcePhase = 0
        toPhase = 0

    # some protective checks
    if sourcePhase in [1, 2, 3] and toPhase in [1, 2, 3] and sourcePhase != toPhase:
        XSecArray[XSecArray == sourcePhase] = toPhase
        print("Phase {} changed to phase {}".format(sourcePhase, toPhase))
        printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)
    else:
        print("cannot swap!")


def showMeGeom(*arg):
    # kicking off the geometry navi Window
    # root = Tk()
    # root.title('Geometry Modifications')
    # geomMod = gui.geometryModWindow(root, canvas=w)
    global XSecArray

    inDialog = gui.MyPtrn(master)
    master.wait_window(inDialog.top)

    try:
        dX = inDialog.dX
        dY = inDialog.dY
        N = inDialog.N
    except:
        dX = 0
        dY = 0
        N = 0

    if dX != 0 or dY != 0 and N >= 1:
        XSecArray = csd.n_cloneGeometry(dX, dY, N, XSecArray)
        print(XSecArray.shape)
        printTheArray(XSecArray, canvas=w)


def vectorizeTheArray(*arg):
    """
    This function analyse the cross section array and returns vector of all set
    (equal to 1) elements. This allows to minimize the size of further calculation
    arrays only to active elements.

    and for te moment do the all math for calculations.
    """
    global \
        elementsVector, \
        resultsArray, \
        resultsCurrentVector, \
        frequency, \
        powerLosses, \
        resultsArrayPower, \
        powerLossesVector

    # Read the setup params from GUI
    setParameters()

    # lets check if there is anything in the xsection geom array
    if np.sum(XSecArray) > 0:
        # We get vectors for each phase`
        elementsVectorPhA = csd.n_arrayVectorize(
            inputArray=XSecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm
        )
        elementsVectorPhB = csd.n_arrayVectorize(
            inputArray=XSecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm
        )
        elementsVectorPhC = csd.n_arrayVectorize(
            inputArray=XSecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm
        )
        # From here is the rest of calulations
        perymeterA = csd.n_perymiter(elementsVectorPhA, XSecArray, dXmm, dYmm)
        perymeterB = csd.n_perymiter(elementsVectorPhB, XSecArray, dXmm, dYmm)
        perymeterC = csd.n_perymiter(elementsVectorPhC, XSecArray, dXmm, dYmm)

        # memorize the number of elements in each phase
        elementsPhaseA = elementsVectorPhA.shape[0]
        elementsPhaseB = elementsVectorPhB.shape[0]
        elementsPhaseC = elementsVectorPhC.shape[0]

        # Lets put the all phases togethrt
        if elementsPhaseA != 0 and elementsPhaseB != 0 and elementsPhaseC != 0:
            elementsVector = np.concatenate(
                (elementsVectorPhA, elementsVectorPhB, elementsVectorPhC), axis=0
            )

        elif elementsPhaseA == 0:
            if elementsPhaseB == 0:
                elementsVector = elementsVectorPhC
            elif elementsPhaseC == 0:
                elementsVector = elementsVectorPhB
            else:
                elementsVector = np.concatenate(
                    (elementsVectorPhB, elementsVectorPhC), axis=0
                )
        else:
            if elementsPhaseB == 0 and elementsPhaseC == 0:
                elementsVector = elementsVectorPhA
            elif elementsPhaseC == 0:
                elementsVector = np.concatenate(
                    (elementsVectorPhA, elementsVectorPhB), axis=0
                )
            else:
                elementsVector = np.concatenate(
                    (elementsVectorPhA, elementsVectorPhC), axis=0
                )

        print(elementsVector.shape)

        admitanceMatrix = np.linalg.inv(
            csd.n_getImpedanceArray(
                csd.n_getDistancesArray(elementsVector),
                freq=frequency,
                dXmm=dXmm,
                dYmm=dYmm,
                temperature=temperature,
            )
        )

        # Let's put here some voltage vector
        vA = np.ones(elementsPhaseA)
        vB = np.ones(elementsPhaseB) * (-0.5 + (np.sqrt(3) / 2) * 1j)
        vC = np.ones(elementsPhaseC) * (-0.5 - (np.sqrt(3) / 2) * 1j)

        voltageVector = np.concatenate((vA, vB, vC), axis=0)

        # Lets calculate the currebt vector as U = ZI >> Z^-1 U = I
        # and Y = Z^-1
        # so finally I = YU - as matrix multiplication goes

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        # And now we need to get solution for each phase to normalize it
        currentPhA = currentVector[0:elementsPhaseA]
        currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB : 1]
        currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

        # Normalize the solution vectors fr each phase
        currentPhA = currentPhA / csd.n_getComplexModule(np.sum(currentPhA))
        currentPhB = currentPhB / csd.n_getComplexModule(
            np.sum(currentPhB)
        )  # *(-0.5 + (np.sqrt(3)/2)*1j))
        currentPhC = currentPhC / csd.n_getComplexModule(
            np.sum(currentPhC)
        )  # *(-0.5 - (np.sqrt(3)/2)*1j))

        # Print out he results currents in each phase
        print(
            "sumy: "
            + str(csd.n_getComplexModule(np.sum(currentPhA)))
            + " : "
            + str(csd.n_getComplexModule(np.sum(currentPhB)))
            + " : "
            + str(csd.n_getComplexModule(np.sum(currentPhC)))
            + " : "
        )

        print(
            "sumy: "
            + str((np.sum(currentPhA)))
            + " : "
            + str((np.sum(currentPhB)))
            + " : "
            + str((np.sum(currentPhC)))
            + " : "
        )

        print("Current vector:")
        print(currentVector.shape)
        print("Current vector elements module:")
        getMod = np.vectorize(csd.n_getComplexModule)

        resultsCurrentVector = np.concatenate(
            (currentPhA, currentPhB, currentPhC), axis=0
        )

        resultsCurrentVector = getMod(resultsCurrentVector)
        resistanceVector = csd.n_getResistanceArray(
            elementsVector, dXmm=dXmm, dYmm=dYmm, temperature=temperature
        )
        resultsCurrentVector *= curentRMS

        powerLossesVector = resistanceVector * resultsCurrentVector**2
        powerLosses = np.sum(powerLossesVector)

        # Power losses per phase
        powPhA = np.sum(powerLossesVector[0:elementsPhaseA])
        powPhB = np.sum(
            powerLossesVector[elementsPhaseA : elementsPhaseA + elementsPhaseB : 1]
        )
        powPhC = np.sum(powerLossesVector[elementsPhaseA + elementsPhaseB :])

        print(
            "power losses: {} [W] \n phA: {}[W]\n phB: {}[W]\n phC: {}[W]".format(
                powerLosses, powPhA, powPhB, powPhC
            )
        )

        print(
            "Phases perymeters:\nA: {}mm\nB: {}mm\nC: {}mm\n".format(
                perymeterA, perymeterB, perymeterC
            )
        )

        powerLosses = [powerLosses, powPhA, powPhB, powPhC]

        # Converting results to form of density
        powerLossesVector /= dXmm * dYmm

        # Converting resutls to current density
        resultsCurrentVector /= dXmm * dYmm

        # Recreating the solution to form of cross section array
        resultsArray = csd.n_recreateresultsArray(
            elementsVector=elementsVector,
            resultsVector=resultsCurrentVector,
            initialGeometryArray=XSecArray,
        )
        resultsArrayPower = csd.n_recreateresultsArray(
            elementsVector=elementsVector,
            resultsVector=powerLossesVector,
            initialGeometryArray=XSecArray,
        )

        # Showing the results
        showResults()


def drawGeometryArray(theArrayToDisplay):
    global figGeom, geomax, geomim

    title_font = {"size": "11", "color": "black", "weight": "normal"}
    axis_font = {"size": "10"}

    my_cmap = matplotlib.cm.get_cmap("jet")
    my_cmap.set_under("w")

    figGeom = plt.figure(1)

    if np.sum(theArrayToDisplay) == 0:
        vmin = 0
    else:
        vmin = 0.8

    geomax = figGeom.add_subplot(1, 1, 1)

    plotWidth = (theArrayToDisplay.shape[1]) * dXmm
    plotHeight = (theArrayToDisplay.shape[0]) * dYmm

    geomim = geomax.imshow(
        theArrayToDisplay,
        cmap="jet",
        interpolation="none",
        extent=[0, plotWidth, plotHeight, 0],
        vmin=vmin,
    )

    geomax.set_xticks(np.arange(0, plotWidth, 2 * dXmm))
    geomax.set_yticks(np.arange(0, plotHeight, 2 * dYmm))

    figGeom.autofmt_xdate(bottom=0.2, rotation=45, ha="right")

    plt.xlabel("size [mm]", **axis_font)
    plt.ylabel("size [mm]", **axis_font)
    plt.axis("scaled")
    plt.tight_layout()

    plt.grid(True)
    plt.show()


def showResults():
    title_font = {"size": "11", "color": "black", "weight": "normal"}
    axis_font = {"size": "10"}

    if np.sum(resultsArray) != 0:
        # Checking the area in array that is used by geometry to limit the display
        min_row = int(np.min(elementsVector[:, 0]))
        max_row = int(np.max(elementsVector[:, 0]) + 1)

        min_col = int(np.min(elementsVector[:, 1]))
        max_col = int(np.max(elementsVector[:, 1]) + 1)

        # Cutting down results array to the area with geometry
        resultsArrayDisplay = resultsArray[min_row:max_row, min_col:max_col]
        resultsArrayDisplay2 = resultsArrayPower[min_row:max_row, min_col:max_col]

        # Checking out what are the dimensions od the ploted area
        # to make propper scaling

        plotWidth = (resultsArrayDisplay.shape[1]) * dXmm
        plotHeight = (resultsArrayDisplay.shape[0]) * dYmm

        fig = plt.figure("Results Window")
        ax = fig.add_subplot(1, 1, 1)

        my_cmap = matplotlib.cm.get_cmap("jet")
        my_cmap.set_under("w")

        im = ax.imshow(
            resultsArrayDisplay,
            cmap=my_cmap,
            interpolation="none",
            vmin=0.8 * np.min(resultsCurrentVector),
            extent=[0, plotWidth, plotHeight, 0],
        )
        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            label="Current Density [A/mm$^2$]",
            alpha=0.5,
            fraction=0.046,
        )
        plt.axis("scaled")

        ax.set_title(
            str(frequency)
            + "[Hz] / "
            + str(curentRMS)
            + "[A] / "
            + str(temperature)
            + "[$^o$C]\n Power Losses {0[0]:.2f}[W] \n phA: {0[1]:.2f} phB: {0[2]:.2f} phC: {0[3]:.2f}".format(
                powerLosses
            ),
            **title_font,
        )

        plt.xlabel("size [mm]", **axis_font)
        plt.ylabel("size [mm]", **axis_font)

        fig.autofmt_xdate(bottom=0.2, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()
    else:
        print("No results available! Run the analysis first.")


def shiftL():
    """This is just a zero argumet trigger for the geometry shift Button"""
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, -1, 0, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def shiftR():
    """This is just a zero argumet trigger for the geometry shift Button"""
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 1, 0, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def shiftU():
    """This is just a zero argumet trigger for the geometry shift Button"""
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 0, -1, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def shiftD():
    """This is just a zero argumet trigger for the geometry shift Button"""
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 0, 1, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def mainSetup(startSize=3):
    """
    This function set up (or reset) all the main elements
    """
    global \
        temperature, \
        canvas_width, \
        canvas_height, \
        elementsInX, \
        elementsInY, \
        dXmm, \
        dYmm, \
        dX, \
        dY, \
        XSecArray, \
        frequency, \
        resultsArray, \
        curentRMS, \
        globalX, \
        globalY, \
        globalZoom

    globalX = 0
    globalY = 0
    globalZoom = 1

    elementsInX = startSize * 25
    elementsInY = startSize * 25

    dXmm = 10
    dYmm = 10

    dX = canvas_width / elementsInX
    dY = canvas_height / elementsInY

    XSecArray = np.zeros(shape=[elementsInY, elementsInX])
    resultsArray = np.zeros(shape=[elementsInY, elementsInX])

    frequency = 50
    curentRMS = 1000
    temperature = 35


def setParameters(*arg):
    global \
        temperature, \
        frequency, \
        AnalysisFreq, \
        curentRMS, \
        dXmm, \
        dYmm, \
        analysisDX, \
        analysisDY

    dXmm = float(myEntryDx.get())
    dYmm = dXmm

    try:
        forceCalc.dXmm = dXmm
        forceCalc.dYmm = dYmm
    except:
        pass

    try:
        powerCalc.dXmm = dXmm
        powerCalc.dYmm = dYmm
    except:
        pass

    analysisDX.config(text="dx:\n " + str(dXmm) + "[mm]")
    analysisDY.config(text="dy:\n " + str(dYmm) + "[mm]")


def printTheArray(dataArray, canvas):
    """
    This procedure draw the geometry contained array to given canvas.
    usefull for redraw or draw loaded data
    Inputs:
    dataArray -  the array to display on canvas
    canvas - tkinter canvas object

    it's using global variable canvasElements.
    """
    global canvasElements, selectShadowBox, selectEndPoint, selectStartPoint

    # Let's check the size
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    # Now we calculate the propper dX and dY for this array
    canvasHeight = canvas.winfo_height()
    canvasWidth = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY

    dXY = min(dX, dY)

    lineSkip = 1
    if dXY <= 2:
        lineSkip = 5
    elif dXY < 5:
        lineSkip = 2

    startX = (canvasWidth - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    for anyDrawedElement in canvasElements:
        try:
            canvas.delete(anyDrawedElement)
        except:
            print("Error in removing stuff")
            pass
    canvasElements = []

    canvasElements.append(
        canvas.create_rectangle(
            startX,
            startY,
            canvasWidth - startX,
            canvasHeight - startY,
            fill="white",
            outline="gray",
        )
    )

    colorList = ["red", "green", "blue"]

    for Row in range(elementsInY):
        for Col in range(elementsInX):
            theNumber = int(dataArray[Row][Col])
            if theNumber in [1, 2, 3]:
                fillColor = colorList[theNumber - 1]

                canvasElements.append(
                    canvas.create_rectangle(
                        startX + (Col) * dXY,
                        startY + (Row) * dXY,
                        startX + (Col) * dXY + dXY,
                        startY + (Row) * dXY + dXY,
                        fill=fillColor,
                        outline="",
                    )
                )

            # Handling the lines for the grid
            if Row == elementsInY - 1:
                lineFillColor = "gray"
                lineWidth = 1

                if (Col + globalX) % 5 == 0 and lineSkip == 1:
                    lineFillColor = "dim gray"
                    lineWidth = 2

                if Col % lineSkip == 0:
                    canvasElements.append(
                        canvas.create_line(
                            startX + Col * dXY,
                            startY,
                            startX + Col * dXY,
                            canvasHeight - startY,
                            fill=lineFillColor,
                            width=lineWidth,
                        )
                    )

        lineFillColor = "gray"
        lineWidth = 1
        if (Row + globalY) % 5 == 0 and lineSkip == 1:
            lineFillColor = "dim gray"
            lineWidth = 2

        if Row % lineSkip == 0:
            canvasElements.append(
                canvas.create_line(
                    startX,
                    startY + Row * dXY,
                    canvasWidth - startX,
                    startY + Row * dXY,
                    fill=lineFillColor,
                    width=lineWidth,
                )
            )

    # selection rectangle visualisation

    if selectionArray is not None and len(selectionArray) > 0:
        R1 = min(selectEndPoint[0], selectStartPoint[0]) - globalY
        R2 = max(selectEndPoint[0], selectStartPoint[0]) - globalY

        C1 = min(selectEndPoint[1], selectStartPoint[1]) - globalX
        C2 = max(selectEndPoint[1], selectStartPoint[1]) - globalX

        try:
            canvas.delete(selectShadowBox)
        except:
            pass

        selectShadowBox = canvas.create_rectangle(
            startX + (C1) * dXY,
            startY + (R1) * dXY,
            startX + (C2) * dXY,
            startY + (R2) * dXY,
            fill="",
            outline="yellow",
            width=3,
        )


def setPoint(event):
    """Trigger procesdure for GUI action"""
    actualPhase = phase.get()

    if actualPhase < 4:
        setUpPoint(
            event, actualPhase, zoomInArray(XSecArray, globalZoom, globalX, globalY), w
        )
    elif actualPhase == 4:
        startSelection(event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)
    elif actualPhase == 5:
        pasteSelectionAtPoint(
            event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w
        )
    # elif 20 < actualPhase < 24:
    #     # we are in circle mode

    #     placeCircle(event,zoomInArray(XSecArray, globalZoom, globalX, globalY), w)

    #  Plotting on CAD view if exist
    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def resetPoint(event):
    """Trigger procesdure for GUI action"""
    # csd.n_setUpPoint(event, Set=0, dataArray=zoomInArray(XSecArray,globalZoom,globalX,globalY), canvas=w)
    setUpPoint(
        event,
        Set=0,
        dataArray=zoomInArray(XSecArray, globalZoom, globalX, globalY),
        canvas=w,
    )

    #  Plotting on CAD view if exist
    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def setUpPoint(event, Set, dataArray, canvas):
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

    dXY = min(dX, dY)

    startX = (canvasWidth - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    if (
        event.x < canvasWidth - startX
        and event.y < canvasHeight - startY
        and event.x > startX
        and event.y > startY
    ):
        Col = int((event.x - startX) / dXY)
        Row = int((event.y - startY) / dXY)

        if Set in [0, 1, 2, 3]:
            dataArray[Row][Col] = Set

    printTheArray(dataArray, canvas)


def moveShadowPoint(event, dataArray, canvas):
    """
    This procedure make the gray box folloe the mause moves
    """
    global shadowBox
    try:
        canvas.delete(shadowBox)
    except:
        pass

    # gathering some current data
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY

    dXY = min(dX, dY)

    startX = (canvasWidth - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    if (
        event.x < canvasWidth - startX
        and event.y < canvasHeight - startY
        and event.x > startX
        and event.y > startY
    ):
        Col = int((event.x - startX) / dXY)
        Row = int((event.y - startY) / dXY)

        shadowBox = canvas.create_rectangle(
            startX + (Col) * dXY,
            startY + (Row) * dXY,
            startX + (Col) * dXY + dXY,
            startY + (Row) * dXY + dXY,
            fill="gray",
            outline="yellow",
        )


def shadowPoint(event):
    """
    This is a trigger function for moveShadowPoint
    """
    # moveShadowPoint(event, XSecArray, w)
    moveShadowPoint(event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)


def startSelection(event, dataArray, canvas):
    """
    Procedure to start selecting area in the canvas
    and in the dataArray
    """
    global inSelectMode, selectStartPoint, selectEndPoint, selectShadowBox

    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY

    dXY = min(dX, dY)

    startX = (canvasWidth - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    if (
        event.x < canvasWidth - startX
        and event.y < canvasHeight - startY
        and event.x > startX
        and event.y > startY
    ):
        Col = int((event.x - startX) / dXY)
        Row = int((event.y - startY) / dXY)

        if not inSelectMode:
            inSelectMode = True
            selectStartPoint = (Row, Col)

        else:
            selectEndPoint = (Row + 1, Col + 1)

            R1 = min(selectEndPoint[0], selectStartPoint[0])
            R2 = max(selectEndPoint[0], selectStartPoint[0])

            C1 = min(selectEndPoint[1], selectStartPoint[1])
            C2 = max(selectEndPoint[1], selectStartPoint[1])

            try:
                canvas.delete(selectShadowBox)
            except:
                pass

            selectShadowBox = canvas.create_rectangle(
                startX + (C1) * dXY,
                startY + (R1) * dXY,
                startX + (C2) * dXY,
                startY + (R2) * dXY,
                fill="",
                outline="yellow",
                width=3,
            )
            # temporary solution to visualise in developement
            # dataArray[R1:R2, C1:C2] = 3
            # printTheArray(dataArray, canvas)


def endSelection(event):
    """
    procedutr to end the selection process
    """
    global \
        inSelectMode, \
        selectStartPoint, \
        selectEndPoint, \
        selectionMaskArray, \
        selectionArray

    if inSelectMode and selectEndPoint is not None and selectStartPoint is not None:
        inSelectMode = False

        R1 = min(selectEndPoint[0], selectStartPoint[0])
        R2 = max(selectEndPoint[0], selectStartPoint[0])

        C1 = min(selectEndPoint[1], selectStartPoint[1])
        C2 = max(selectEndPoint[1], selectStartPoint[1])

        # selectionArray = np.copy(XSecArray[R1:R2, C1:C2]) # this seems to work only if not zoomed
        # selectionMaskArray = np.empty_like(XSecArray)

        selectionMaskArray = np.empty_like(
            zoomInArray(XSecArray, globalZoom, globalX, globalY)
        )
        selectionMaskArray[R1:R2, C1:C2] = 1

        selectionArray = np.copy(
            zoomInArray(XSecArray, globalZoom, globalX, globalY)[R1:R2, C1:C2]
        )  # this makes it copy right data if in zoom mode.

        # auto switch to paste mode after selection is done
        phase.set(5)

        # development debug
        # print(selectionArray)

        # selectStartPoint = None
        # selectEndPoint = None


def pasteSelectionAtPoint(event, dataArray, canvas):
    """
    Procedure that paste the selectionArray into the
    dataArray at the click position
    """
    global selectionArray, XSecArray

    if selectionArray is not None and len(selectionArray) and not inSelectMode:
        pasteRows = selectionArray.shape[0]
        pasteCols = selectionArray.shape[1]

        elementsInY = dataArray.shape[0]
        elementsInX = dataArray.shape[1]

        canvasHeight = canvas.winfo_height()
        canvasWidth = canvas.winfo_width()

        dX = canvasWidth / elementsInX
        dY = canvasHeight / elementsInY

        dXY = min(dX, dY)

        startX = (canvasWidth - dXY * elementsInX) / 2
        startY = (canvasHeight - dXY * elementsInY) / 2

        if (
            event.x < canvasWidth - startX
            and event.y < canvasHeight - startY
            and event.x > startX
            and event.y > startY
        ):
            Col = int((event.x - startX) / dXY)
            Row = int((event.y - startY) / dXY)

            R1 = Row + globalY
            R2 = R1 + pasteRows
            R2 = min(R2, XSecArray.shape[0])

            C1 = Col + globalX
            C2 = C1 + pasteCols
            C2 = min(C2, XSecArray.shape[1])

            selectedPasteMode = paste_mode.get()

            if selectedPasteMode == 1:
                # take the clipboard if its > 0 else take oryginal data
                XSecArray[R1:R2, C1:C2] = np.where(
                    selectionArray[: R2 - R1, : C2 - C1] > 0,
                    selectionArray[: R2 - R1, : C2 - C1],
                    XSecArray[R1:R2, C1:C2],
                )

            elif selectedPasteMode == 2:
                # overwite all data
                XSecArray[R1:R2, C1:C2] = selectionArray[: R2 - R1, : C2 - C1]

            elif 10 < selectedPasteMode < 20:
                # paste as target phase
                targetPhase = selectedPasteMode - 10
                XSecArray[R1:R2, C1:C2] = np.where(
                    selectionArray[: R2 - R1, : C2 - C1] > 0,
                    targetPhase,
                    XSecArray[R1:R2, C1:C2],
                )

            printTheArray(dataArray, canvas)

            # if this is active we drop the clipboard data at paste.
            # selectionArray = None


def InterCode():
    """This is internal sudo-code inerpreter for easier geometry creation
    Available commands:
    C(x,y,D1,Ph1) - circle at x,y with diameter D1, set as phase [mm]
    C(x,y,D1,Ph1,D2,Ph2) - torus at x, y with up to D2 as Ph2 and above to D1 as ph1
    R(x,y,W,H) - rectangle x,y left top, W width, H height - [mm]
    """

    codeLines = text_input.get("1.0", END).split("\n")
    codeSteps, *_ = ic.textToCode(codeLines)

    if codeSteps:
        for step in codeSteps:
            step[0](*step[1], XSecArray=XSecArray, dXmm=dXmm)

    redraw()


def getCanvas():
    """This functoion is to determine the best parameters for the canvas
    based on the given geometry steps defined by the inner code."""

    codeLines = text_input.get("1.0", END).split("\n")
    codeSteps, *_ = ic.textToCode(codeLines)

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

        print(X)
        print(Y)
        print(f"Dimention range: {min(X)}:{max(X)}; {min(Y)}:{max(Y)}")
        size = (max(X) - min(X), max(Y) - min(Y))
        print(size)

        # I have no good idea how to figure out the best cell size
        # so for now it's just some stuff..
        if circles:
            sizes = [4, 2.5, 2, 1]
        else:
            sizes = [10, 5, 4, 2.5, 2, 1]
        for xd in sizes:
            if (size[0] % xd == 0) and (size[1] % xd == 0):
                break
        print(f"The dx: {xd}mm")

        elements = int(max(size[0] / xd, size[1] / xd))
        print(f"Canvas elements neede: {elements}")

        global dXmm, dYmm, XSecArray
        dXmm = dYmm = xd
        XSecArray = np.zeros([elements, elements])

        for step in codeSteps:
            step[0](*step[1], shift=(min(X), min(Y)), XSecArray=XSecArray, dXmm=dXmm)

    myEntryDx.delete(0, END)
    myEntryDx.insert(END, str(dXmm))
    redraw()


######## End of functions definition ############


master = Tk()
master.title("Cross Section Designer")

img = PhotoImage(file="CSDico.gif")
master.tk.call("wm", "iconphoto", master._w, img)

canvas_width = 650
canvas_height = 650

# master.resizable(width=False, height=False)
master.resizable(width=not False, height=not False)

mainSetup()

# bindind the resize event
master.bind("<Configure>", redraw)

w = Canvas(master, width=canvas_width, height=canvas_height)
w.configure(background="gray69")
w.grid(row=1, column=1, columnspan=5, rowspan=25, sticky=W + E + N + S, padx=1, pady=1)

canvasElements = []
shadowBox = None

inSelectMode = False
selectStartPoint = None
selectEndPoint = None
selectShadowBox = None

selectionMaskArray = None
selectionArray = None


# the menu bar stuff
menu_bar = Menu(master)

file_menu = Menu(menu_bar)
file_menu.add_command(label="New geometry", command=clearArrayAndDisplay)
file_menu.add_separator()
file_menu.add_command(label="Load from file", command=loadArrayFromFile)
file_menu.add_command(label="Import from picture", command=importArrayFromPicture)
file_menu.add_separator()
file_menu.add_command(label="Save to file", command=saveArrayToFile)
menu_bar.add_cascade(label="File", menu=file_menu)

analyze_menu = Menu(menu_bar)
analyze_menu.add_command(label="Power Losses ProSolver", command=showMePro)
analyze_menu.add_command(label="Electro Dynamic Forces", command=showMeForces)
analyze_menu.add_command(label="Equivalent Impedance Model", command=showMeZ)
analyze_menu.add_command(label="Equivalent Impedance Model 3f Shunt", command=showMeZ3f)
menu_bar.add_cascade(label="Analyze...", menu=analyze_menu)

geometry_menu = Menu(menu_bar)
geometry_menu.add_command(label="Pattern", command=showMeGeom)
geometry_menu.add_command(label="Swap", command=showReplacer)
geometry_menu.add_separator()
geometry_menu.add_command(label="Subdivide(+)", command=subdivideArray)
geometry_menu.add_command(label="Simplify(-)", command=simplifyArray)
geometry_menu.add_separator()
geometry_menu.add_command(label="Extend Canvas", command=extendArray)
menu_bar.add_cascade(label="Geometry", menu=geometry_menu)

view_menu = Menu(menu_bar)
view_menu.add_command(label="Open CAD view window", command=displayArrayAsImage)
menu_bar.add_cascade(label="View", menu=view_menu)

master.config(menu=menu_bar)

# tools selector bar
# phase selection pane
A_icon_white = PhotoImage(file="csdicons/A_white.png")
B_icon_white = PhotoImage(file="csdicons/B_white.png")
C_icon_white = PhotoImage(file="csdicons/C_white.png")
cut_icon_white = PhotoImage(file="csdicons/cut_white.png")
select_icon_white = PhotoImage(file="csdicons/select_white.png")
paste_icon_white = PhotoImage(file="csdicons/paste_white.png")
paste_icon_all = PhotoImage(file="csdicons/paste_full.png")
paste_icon_A = PhotoImage(file="csdicons/paste_A.png")
paste_icon_B = PhotoImage(file="csdicons/paste_B.png")
paste_icon_C = PhotoImage(file="csdicons/paste_C.png")


phase_frame = LabelFrame(master, text="Active operation")
phase_frame.grid(row=1, column=8, columnspan=3)

phase = IntVar()
phase.set(1)  # initialize

Btn = Radiobutton(
    phase_frame,
    image=A_icon_white,
    variable=phase,
    value=1,
    indicatoron=0,
    height=32,
    width=32,
    bg="red",
    highlightbackground="red",
)
Btn.grid(row=0, column=0, padx=1, pady=2)

Btn = Radiobutton(
    phase_frame,
    image=B_icon_white,
    variable=phase,
    value=2,
    indicatoron=0,
    height=32,
    width=32,
    bg="green",
    highlightbackground="green",
)
Btn.grid(row=0, column=1, padx=1, pady=2)

Btn = Radiobutton(
    phase_frame,
    image=C_icon_white,
    variable=phase,
    value=3,
    indicatoron=0,
    height=32,
    width=32,
    bg="blue",
    highlightbackground="blue",
)
Btn.grid(row=0, column=2, padx=1, pady=2)

Btn = Radiobutton(
    phase_frame,
    image=cut_icon_white,
    variable=phase,
    value=0,
    indicatoron=0,
    height=32,
    width=32,
    bg="gray",
    highlightbackground="gray",
)
Btn.grid(row=1, column=0, padx=1, pady=2)

Btn = Radiobutton(
    phase_frame,
    image=select_icon_white,
    variable=phase,
    value=4,
    indicatoron=0,
    height=32,
    width=32,
    bg="gray",
    highlightbackground="gray",
)
Btn.grid(row=1, column=1, padx=1, pady=2)

Btn = Radiobutton(
    phase_frame,
    image=paste_icon_white,
    variable=phase,
    value=5,
    indicatoron=0,
    height=32,
    width=32,
    bg="gray",
    highlightbackground="gray",
)
Btn.grid(row=1, column=2, padx=1, pady=2)

# paste mode frame
paste_frame = LabelFrame(master, text="Paste mode")
paste_frame.grid(row=3, column=8, columnspan=3)

paste_mode = IntVar()
paste_mode.set(1)

Btn = Radiobutton(
    paste_frame,
    image=paste_icon_white,
    variable=paste_mode,
    value=1,
    indicatoron=0,
    height=32,
    width=32,
    bg="gray",
    highlightbackground="dark gray",
)
Btn.grid(row=0, column=0, padx=1, pady=1)

Btn = Radiobutton(
    paste_frame,
    image=paste_icon_all,
    variable=paste_mode,
    value=2,
    indicatoron=0,
    height=32,
    width=32,
    bg="gray",
    highlightbackground="dark gray",
)
Btn.grid(row=0, column=1, padx=1, pady=1)

Btn = Radiobutton(
    paste_frame,
    image=paste_icon_A,
    variable=paste_mode,
    value=11,
    indicatoron=0,
    height=32,
    width=32,
    bg="red",
    highlightbackground="red",
)
Btn.grid(row=1, column=0, padx=1, pady=2)

Btn = Radiobutton(
    paste_frame,
    image=paste_icon_B,
    variable=paste_mode,
    value=12,
    indicatoron=0,
    height=32,
    width=32,
    bg="green",
    highlightbackground="green",
)
Btn.grid(row=1, column=1, padx=1, pady=2)

Btn = Radiobutton(
    paste_frame,
    image=paste_icon_C,
    variable=paste_mode,
    value=13,
    indicatoron=0,
    height=32,
    width=32,
    bg="blue",
    highlightbackground="blue",
)
Btn.grid(row=1, column=2, padx=1, pady=2)


# geometry modyfication pane
up_icon_white = PhotoImage(file="csdicons/up_white.png")
down_icon_white = PhotoImage(file="csdicons/down_white.png")
left_icon_white = PhotoImage(file="csdicons/left_white.png")
right_icon_white = PhotoImage(file="csdicons/right_white.png")

mod_frame = LabelFrame(master, text="Shift sel. phase")
mod_frame.grid(row=6, column=8, columnspan=3, sticky="S")

# geometry shift cross navi
print_button_zoom = Button(
    mod_frame,
    image=left_icon_white,
    width=24,
    height=24,
    relief=FLAT,
    command=shiftL,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=1, column=0, padx=5, pady=5)
print_button_zoom = Button(
    mod_frame,
    image=right_icon_white,
    width=24,
    height=24,
    relief=FLAT,
    command=shiftR,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=1, column=2, padx=5, pady=5)
print_button_zoom = Button(
    mod_frame,
    image=up_icon_white,
    width=24,
    height=24,
    relief=FLAT,
    command=shiftU,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=0, column=1, padx=5, pady=5)
print_button_zoom = Button(
    mod_frame,
    image=down_icon_white,
    width=24,
    height=24,
    relief=FLAT,
    command=shiftD,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=1, column=1, padx=5, pady=5)


emptyOpis = Label(text="", height=3)
emptyOpis.grid(
    row=5,
    column=0,
)

analysisDX = Label(text="grid:", height=2)
analysisDX.grid(row=5, column=8, columnspan=1)
analysisDY = Label(text="[mm]", height=2)
analysisDY.grid(row=5, column=10, columnspan=1)

myEntryDx = Entry(master, width=5)
myEntryDx.insert(END, str(dXmm))
myEntryDx.grid(row=5, column=9, columnspan=1, padx=1, pady=1)
myEntryDx.bind("<Return>", setParameters)
myEntryDx.bind("<FocusOut>", setParameters)


# Geometry navigation frame
zoom_in_icon = PhotoImage(file="csdicons/zoomin.png")
zoom_out_icon = PhotoImage(file="csdicons/zoomout.png")
up_icon = PhotoImage(file="csdicons/up.png")
down_icon = PhotoImage(file="csdicons/down.png")
left_icon = PhotoImage(file="csdicons/left.png")
right_icon = PhotoImage(file="csdicons/right.png")

navi_frame = LabelFrame(master, text="View navi")
navi_frame.grid(row=21, column=8, columnspan=3, sticky="S")

print_button_zoom = Button(
    navi_frame, image=zoom_in_icon, width=32, height=32, relief=FLAT, command=zoomIn
)
print_button_zoom.grid(row=0, column=0, padx=5, pady=5, columnspan=1)
print_button_zoom = Button(
    navi_frame, image=zoom_out_icon, width=32, height=32, relief=FLAT, command=zoomOut
)
print_button_zoom.grid(row=0, column=2, padx=5, pady=5, columnspan=1)


# first cross navi
print_button_zoom = Button(
    navi_frame,
    image=left_icon,
    width=24,
    height=24,
    relief=FLAT,
    command=zoomL,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=1, column=0, padx=5, pady=5)
print_button_zoom = Button(
    navi_frame,
    image=right_icon,
    width=24,
    height=24,
    relief=FLAT,
    command=zoomR,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=1, column=2, padx=5, pady=5)
print_button_zoom = Button(
    navi_frame,
    image=up_icon,
    width=24,
    height=24,
    relief=FLAT,
    command=zoomU,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=0, column=1, padx=5, pady=5)
print_button_zoom = Button(
    navi_frame,
    image=down_icon,
    width=24,
    height=24,
    relief=FLAT,
    command=zoomD,
    repeatdelay=100,
    repeatinterval=100,
)
print_button_zoom.grid(row=1, column=1, padx=5, pady=5)

# advanced geometry triggers
code_frame = LabelFrame(master, text="inner-code window")
code_frame.grid(row=1, column=11, rowspan=25, columnspan=3, sticky="N", padx=5, pady=0)

text_input = Text(code_frame, height=20, width=25)
text_input.grid(row=1, column=1, columnspan=3, padx=10, pady=10)
btn = Button(code_frame, text="Execute InnerCode as is", command=InterCode)
btn.grid(row=19, column=1, columnspan=3)
btn = Button(code_frame, text="Create new geometry", command=getCanvas)
btn.grid(row=20, column=1, columnspan=3)


w.bind("<Button 1>", setPoint)
w.bind("<Button 3>", resetPoint)
w.bind("<B1-Motion>", setPoint)
w.bind("<B3-Motion>", resetPoint)
w.bind("<Motion>", shadowPoint)
w.bind("<ButtonRelease-1>", endSelection)

w.bind("<Button 2>", showXsecArray)

w.bind("<Left>", zoomL)
w.bind("<Right>", zoomR)
w.bind("<Up>", zoomU)
w.bind("<Down>", zoomD)

message = Label(master, text="use: Left Mouse Button to Set conductor, Right to reset")
message.grid(row=26, column=1, columnspan=3)

# rescaling behaviour
master.grid_rowconfigure(0, weight=0)
master.grid_rowconfigure(12, weight=1)
master.grid_columnconfigure(0, weight=0)
master.grid_columnconfigure(1, weight=1)

master.update()

canvas_height = w.winfo_height()
canvas_width = w.winfo_width()

printTheArray(dataArray=XSecArray, canvas=w)


# print(phase)

mainloop()
