import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
import os.path
import time
import argparse
import sys

# Importing local library
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2
from csdlib import csdgui as gui
from csdlib import csdcli as cli
from csdlib import innercode as ic
from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm
from csdlib import csdsolve as csds
from csdlib import csdos


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
        
        config = {'frequency': frequency, 'temperature': temperature, 'length': 1000, 'htc': 5, 'conductivity': 56e6, 'temRcoeff': 3.9e-3, 'material': -1, 'simple': False, 'markdown': False, 'csv': False, 'verbose': False, 'draw': False, 'results': True, 'bars': False, 'bardetails': False, 'geometry': '', 'current': curentRMS}
        
        verbose = config["verbose"]
        simple = config["simple"]
        csv = config["csv"]

        # for simplicity so the log procedure can see it globally
        csd.verbose = verbose

        list_of_phases = np.unique(XSecArray).astype(int)
        list_of_phases = [int(n) for n in list_of_phases if n != 0]
        original_phase_index = {index: phase for index, phase in enumerate(list_of_phases)} 
        new_phase_index = {phase: index for index, phase in enumerate(list_of_phases)} 
        # normalizing the phases numbering
        normalized_XsecArr = np.zeros(XSecArray.shape)
        for index,phase in enumerate([0]+list_of_phases):
            if phase != 0:
                normalized_XsecArr[XSecArray==phase]=index
        XSecArray = normalized_XsecArr  

        number_of_phases = len(list_of_phases)
        # will create this dict - just to don't modify the downhill code yet.
        phase_index = {index: index for index, phase in enumerate(list_of_phases)}
        
        Irms = config["current"]
        # Current vector
        Icurrent = []
        phi = [120, 0, 240, 120, 0, 240]
        direction = [0, 0, 0, 180, 180, 180]
        x = 0
        for n in range(number_of_phases - 1):
            Icurrent.append((Irms, phi[x] + direction[x]))
            x += 1
            if x > len(phi):
                x = 0
        
        f = config["frequency"]
        length = config["length"]
        t = config["temperature"]
        HTC = config["htc"]
        
        # Reading Material data
        M_list = csdos.read_file_to_list("setup/materials.txt")[1:]
        if M_list:
            MaterialsDB = csdos.get_material_from_list(M_list)

        phases_material = [0 for _ in range(number_of_phases)]
        
        this_material = csdos.Material(
            "Cu", config["conductivity"], config["temRcoeff"], 0, 0
        )
        
        if this_material:
            sigma = this_material.sigma
            r20 = this_material.alpha
            phases_material = [this_material]
            
        (
            resultsCurrentVector,
            powerResults,
            elementsVector,
            powerLossesSolution,
            complexCurrent,
            vPh,
            mi_r_weighted
        ) = csds.solve_with_magnetic(
            XsecArr=XSecArray,
            phases_materials=phases_material,
            dXmm=dXmm,
            dYmm=dYmm,
            currents=Icurrent,
            frequency=f,
            length=length,
            temperature=t,
            verbose=verbose,
        )

        powerLosses, powPh = powerResults
        
        # Recreating the solution to form of cross section array
        resultsArray = csd.n_recreateresultsArray(
            elementsVector=elementsVector,
            resultsVector=resultsCurrentVector,
            initialGeometryArray=XSecArray,
        )
        resultsArrayPower = csd.n_recreateresultsArray(
            elementsVector=elementsVector,
            resultsVector=powerLossesSolution,
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


def get_cli_args():
    """
    Handling the cli line parameters.
    """
    parser = argparse.ArgumentParser(
        description="CSD cli executor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--size", help="Max single cell size in [mm]", type=float, default=5
    )
    parser.add_argument(
        "-f", "--frequency", type=float, default=50.0, help="Currents frequency in Hz"
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=140.0,
        help="Conductors temperature in deg C",
    )
    parser.add_argument(
        "-l", "--length", type=float, default=1000.0, help="Analyzed length"
    )
    parser.add_argument(
        "-htc",
        "--htc",
        type=float,
        default=5,
        help="Heat transfer coefficient for cooling of conductors in [W/m.K]",
    )
    parser.add_argument(
        "-sig",
        "--conductivity",
        type=float,
        default=56.0e6,
        help="Conductors conductivity at 20 degC in [S]",
    )
    parser.add_argument(
        "-rco",
        "--temRcoeff",
        type=float,
        default=3.9e-3,
        help="temperature coeff. of resistnace [1/K]",
    )
    parser.add_argument(
        "-mat",
        "--material",
        type=int,
        default=-1,
        help="Material number from the material list (in config directory)",
    )
    (
        parser.add_argument(
            "-sp", "--simple", action="store_true", help="Show only simple output"
        ),
        parser.add_argument(
            "-md",
            "--markdown",
            action="store_true",
            help="Results for bars as markdown table",
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
        parser.add_argument(
            "-b",
            "--bars",
            action="store_true",
            help="Execute the detections of particular conductors.",
        ),
        parser.add_argument(
            "-bd",
            "--bardetails",
            action="store_true",
            help="If bars - the this define if to show results on plot.",
        ),
    )

    parser.add_argument("geometry", help="Geometry description file in .csd format")
    parser.add_argument(
        "current",
        help="Current RMS value for the 3 phase \
                symmetrical analysis in ampers [A]",
        type=float,
    )

    args = parser.parse_args()
    return vars(args)


def run_cli():
    """
    This is the place where the main flow of operation is carried.
    """

    config = get_cli_args()
    verbose = config["verbose"]
    simple = config["simple"]
    csv = config["csv"]

    # for simplicity so the log procedure can see it globally
    csdf.verbose = verbose

    csdf.myLog()
    csdf.myLog("Starting operations...")
    csdf.myLog()

    if config["draw"] or config["results"]:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import ListedColormap, BoundaryNorm

    XSecArray = np.zeros((0, 0))
    dXmm = dYmm = 1

    # 2 loading the geometry  and other data:
    XSecArray, dXmm, dYmm, currents, materials = csdf.loadTheData(config["geometry"])

    csdf.myLog("Initial geometry array parameters:")
    csdf.myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    csdf.myLog(f"Data table size: {XSecArray.shape}")
    csdf.myLog(f"Currents definition: {currents}")
    csdf.myLog(f"Material definition pattern: {materials}")

    list_of_phases = np.unique(XSecArray).astype(int)
    list_of_phases = [int(n) for n in list_of_phases if n != 0]
    original_phase_index = {index: phase for index, phase in enumerate(list_of_phases)} 
    new_phase_index = {phase: index for index, phase in enumerate(list_of_phases)} 
    # normalizing the phases numbering
    normalized_XsecArr = np.zeros(XSecArray.shape)
    for index,phase in enumerate([0]+list_of_phases):
        if phase != 0:
            normalized_XsecArr[XSecArray==phase]=index
    XSecArray = normalized_XsecArr  

    number_of_phases = len(list_of_phases)
    # will create this dict - just to don't modify the downhill code yet.
    phase_index = {index: index for index, phase in enumerate(list_of_phases)}

    csdf.myLog(f"phases: {number_of_phases} | {list_of_phases=}")
    csdf.myLog(f"phases: {phase_index=}")

    while dXmm > config["size"]:
        csdf.myLog("Splitting the geometry cells...", end="")
        XSecArray = csdm.arraySlicer(inputArray=XSecArray, subDivisions=2)
        dXmm = dXmm / 2
        dYmm = dYmm / 2

    csdf.myLog()
    csdf.myLog("Adjusted geometry array parameters:")
    csdf.myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    csdf.myLog(f"Data table size: {XSecArray.shape}")

    if config["draw"]:
        # making the draw of the geometry in initial state.

        all_colors = [
            "white",
            "red",
            "green",
            "blue",
            "crimson",
            "blueviolet",
            "yellow",
            "azure",
            "cyan",
            "darksalmon",
        ]

        colors = [all_colors[new_phase_index[n] % len(all_colors)] for n in list_of_phases]

        # Create a custom colormap
        cmap = ListedColormap(colors)

        ax = plt.gca()
        csdf.plot_the_geometry(XSecArray, ax, cmap, dXmm=dXmm, dYmm=dYmm, norm=None)
        plt.show()

        question = input("Do you want to run the analysis? [y]/[n]")
        if question.lower() in ["n", "no", "break", "stop"]:
            sys.exit(0)

    # 3 preparing the solution
    Irms = config["current"]
    # Current vector
    if len(currents) == number_of_phases :
        Icurrent = [[0, 0] for _ in range(number_of_phases)]
        for i in currents:
            p = new_phase_index[int(i[0])]
            Icurrent[p] = [float(i[1]), float(i[2]) + float(i[3])]
    else:
        Icurrent = []
        phi = [120, 0, 240, 120, 0, 240]
        direction = [0, 0, 0, 180, 180, 180]
        x = 0
        for n in range(number_of_phases - 1):
            Icurrent.append((Irms, phi[x] + direction[x]))
            x += 1
            if x > len(phi):
                x = 0

    f = config["frequency"]
    length = config["length"]
    t = config["temperature"]
    HTC = config["htc"]

    # Reading Material data
    M_list = csdos.read_file_to_list("setup/materials.txt")[1:]
    if M_list:
        MaterialsDB = csdos.get_material_from_list(M_list)
        csdf.myLog(f"Materials are: \n {MaterialsDB}")

    phases_material = [0 for _ in range(number_of_phases)]
    # if len(materials) == number_of_phases - 1:
    if len(materials) == number_of_phases:
        # [phase, mat_number]
        for m in materials:
            index = new_phase_index[int(m[0])] 
            index_m = int(m[1])
            if number_of_phases < index or index < 0:
                csdf.myLog("Error! Defined materials for not existing phases!")
                # print("Error! Defined materials for not existing phases!")
                sys.exit(1)
            if len(MaterialsDB) < index_m or index_m < 0:
                csdf.myLog("Error! Defined material not id Materials DB file!")
                # print("Error! Defined material not id Materials DB file!")
                sys.exit(1)

            phases_material[index] = MaterialsDB[index_m]
            this_material = None

    elif config["material"] >= 0:
        # reading the material file and select the material
        if config["material"] < len(MaterialsDB):
            this_material = MaterialsDB[config["material"]]

    else:
        this_material = csdos.Material(
            "Cu", config["conductivity"], config["temRcoeff"], 0, 0
        )

    if this_material:
        csdf.myLog(f"Using material: {this_material.name}")
        sigma = this_material.sigma
        r20 = this_material.alpha
        phases_material = [this_material]

    csdf.myLog()
    csdf.myLog("Starting solver for")
    csdf.myLog(f"{phases_material}")

    csdf.myLog()
    csdf.myLog("Complex form:")

    (
        resultsCurrentVector,
        powerResults,
        elementsVector,
        powerLossesSolution,
        complexCurrent,
        vPh,
        mi_r_weighted
    ) = csds.solve_with_magnetic(
        XsecArr=XSecArray,
        phases_materials=phases_material,
        dXmm=dXmm,
        dYmm=dYmm,
        currents=Icurrent,
        frequency=f,
        length=length,
        temperature=t,
        verbose=verbose,
    )

    powerLosses, powPh = powerResults


    if config["bars"]:
        currentsDraw = csdm.recreateresultsArray(
            elementsVector, complexCurrent, XSecArray, dtype=complex
        )
        powerDraw = csdm.recreateresultsArray(
            elementsVector, powerLossesSolution, XSecArray
        )

        conductorsXsecArr, total_conductors, phases_conductors = csdf.getConductors(
            XSecArray, vPh
        )

        if config["draw"]:

            # making the draw of the geometry in initial state.

            base_cmap = plt.get_cmap("jet", 256)
            colors = base_cmap(np.arange(256))
            colors[0] = [1, 1, 1, 1]
            cmap = ListedColormap(colors)
            norm = plt.Normalize(vmin=0, vmax=total_conductors)

            ax = plt.gca()
            csdf.plot_the_geometry(
                conductorsXsecArr,
                ax,
                cmap,
                dXmm=dXmm,
                dYmm=dYmm,
                norm=norm
            )
            plt.show()

            # just to check
            norm = plt.Normalize(vmin=0, vmax=100)
            ax = plt.gca()
            csdf.plot_the_geometry(
                mi_r_weighted,
                ax,
                cmap,
                dXmm=dXmm,
                dYmm=dYmm,
                norm=norm
            )
            plt.show()

        bars_data = []
        for b in range(1, total_conductors + 1):
            temp_bar_obj = csdf.the_bar()
            temp_bar_obj.elements = csdm.arrayVectorize(
                conductorsXsecArr, phaseNumber=b, dXmm=dXmm, dYmm=dYmm
            )
            coordinateX = sum([x[2] for x in temp_bar_obj.elements]) / len(
                temp_bar_obj.elements
            )
            coordinateY = sum([x[3] for x in temp_bar_obj.elements]) / len(
                temp_bar_obj.elements
            )

            csdf.myLog(
                f"Building bar {b} for {dXmm=} {dYmm=} center: {coordinateX}:{coordinateY} elements: {len(temp_bar_obj.elements)}"
            )
            bars_data.append(temp_bar_obj)

        for i, phase in enumerate(phases_conductors):
            for b in phase:
                bars_data[b - 1].phase = i
                bars_data[b - 1].material = phases_material[i]

        for i, bar in enumerate(bars_data):

            bar.number = i
            bar.perymiter = csdf.getPerymiter(bar.elements, XSecArray, dXmm, dYmm)

            x = y = 0
            for element in bar.elements:
                R = int(element[0])
                C = int(element[1])

                bar.current += currentsDraw[R, C]
                bar.power += powerDraw[R, C]

                x += element[2]
                y += element[3]
            bar.center = [x / len(bar.elements), y / len(bar.elements)]
            bar.length = length
            bar.xsection = len(bar.elements) * dXmm * dYmm

            bar.Rth = bar.length *1e-3 / (bar.xsection*1e-6 * bar.material.thermal_conductivity)
            bar.R = bar.length *1e-3 / (bar.xsection*1e-6 * bar.material.sigma)




        # rebasing phases numbers in bars for the original ones:
        csdf.myLog(original_phase_index)
        for bar in bars_data:
            bar.phase = original_phase_index[bar.phase]
        
        csds.solve_thermal_for_bars(bars_data,HTC=HTC)
        temperature_array = csdf.recreate_temperature_array(bars_data,XSecArray.shape)


    # Results of power losses
    if not simple and not csv:
        print()
        print(
            "--------------------------------------------------------------------------------------"
        )
        print("Results of power losses")
        print(f"\tgeometry: {config['geometry']}")
        print(f"\tMaterials:\n\t{[m.name for m in phases_material]}")
        print(
            f"\tCurrents\n\t{Icurrent}[A,deg]\n\tf={f}[Hz], l={length}[mm], T={t}[degC]"
        )
        print(
            "--------------------------------------------------------------------------------------"
        )

        text_line = "Sum [W]\t| "
        for i, dP in enumerate(powPh):
            text_line += f"dP{i} [W]\t| "
        print(text_line)

        text_line = f"{powerLosses:>6.2f}\t| "
        for i, dP in enumerate(powPh):
            text_line += f"{dP:>6.2f}\t| "
        print(text_line)
        print(
            "--------------------------------------------------------------------------------------"
        )

        if config["bars"]:
            phase_currents = []
            for i, bars in enumerate(phases_conductors):
                print(f"Phase {i}:")
                phase_curr = 0

                for b in bars:
                    bar = bars_data[b - 1]
                    print(
                        f"\t{bar.current=:.2f} {bar.power=:.2f} {bar.perymiter=:.1f} {bar.center=}"
                    )
                    phase_curr += bar.current

                phase_currents.append(phase_curr)

                print(
                    f"\tPhase {i} current sum {phase_curr} / {csdm.getComplexModule(phase_curr)}"
                )

    elif not csv:
        print(f"{f}[Hz] \t {powerLosses:.2f} [W]")

        if config["bars"]:
            for i, bars in enumerate(phases_conductors):
                print(f"Phase {i}: ")
                phase_curr = 0

                for b in bars:
                    bar = bars_data[b - 1]
                    print(
                        f"\t{bar.number:>2}\t{csdm.getComplexModule(bar.current):>8.2f}[A]\t{bar.power:>7.2f}[W]\t{bar.perymiter:>7.2f}[mm]\t{bar.dT:>5.1f}[K]\t{bar.material.name}"
                    )
                    phase_curr += bar.current
    else:
        if config["bars"]:
            if config["markdown"]:
                print(
                    f"phase | bar | TG | Bar Current [A] | Bar dP[W] | Bar Perymetr[mm] | Bar Center X[mm] | Bar Center Y[mm]"
                )
                print(f"---|---|---|---|---|---|---|---")
                for bar in bars_data:
                    print(
                        f"{bar.phase}|{bar.number:>2}|{bar.thermal_group:>2}|{csdm.getComplexModule(bar.current):>8.2f}|{bar.power:>7.2f}|{bar.perymiter:>7.2f}|{bar.center[0]}|{bar.center[1]}"
                    )
            else:
                print(
                    f"phase ; bar ; Bar Current [A] ; Bar dP[W] ; Bar Perymetr[mm] ; Bar Center X[mm] ; Bar Center Y[mm]"
                )
                for bar in bars_data:
                    print(
                        f"{bar.phase};{bar.number:>2};{csdm.getComplexModule(bar.current):>8.2f};{bar.power:>7.2f};{bar.perymiter:>7.2f};{bar.center[0]};{bar.center[1]}"
                    )
        else:
            print(f"{f},{powerLosses:.2f}")

    if config["results"]:
        # getting the current density
        resultsCurrentVector *= 1 / (dXmm * dYmm)
        currentsDraw = csdm.recreateresultsArray(
            elementsVector, resultsCurrentVector, XSecArray
        )
        maxCurrent = resultsCurrentVector.max()
        minCurrent = resultsCurrentVector.min()
        # min_to_draw = maxCurrent/250
        min_to_draw = minCurrent * 0.9

        if 1:
            # making the draw of the geometry in initial state.
            base_cmap = plt.get_cmap("jet", 256)
            colors = base_cmap(np.arange(256))
            colors[0] = [1, 1, 1, 1]
            cmap = ListedColormap(colors)
            norm = plt.Normalize(vmin=min_to_draw, vmax=maxCurrent)

            # Adjust the ticks
            fig = plt.figure()
            if config['bars']:
                gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            else:
                gs = gridspec.GridSpec(1, 2, width_ratios=[80, 20])

            ax = plt.subplot(gs[0])
            bx = plt.subplot(gs[1])
            

            if config['bars']:
                mapx=ax
                norm_t = plt.Normalize(vmin=temperature_array.min(), vmax=temperature_array.max())
                cbx = csdf.plot_the_geometry(temperature_array,bx,cmap,dXmm=dXmm,dYmm=dYmm,norm=norm_t)
                cbar = plt.colorbar(cbx, ax=bx)
            else:
                bx.axis("off")
                mapx = bx

            cax = csdf.plot_the_geometry(
                currentsDraw,
                ax,
                cmap,
                dXmm=dXmm,
                dYmm=dYmm,
                norm=norm
            )


            # Add a color bar
            cbar = plt.colorbar(cax, ax=mapx)
            cbar.set_label("Current density [A/mm2]", rotation=270, labelpad=20)

            text_line = ""
            for i, dP in enumerate(powPh):
                text_line += f"dP{i}:{dP:.2f}[W] "

            ax.set_title(
                f"I={config['current']}A, f={f}Hz, l={length}mm, Temp={t}degC\n\
                total dP = {powerLosses:.2f}[W]\n\
                {text_line}\n\
                Current Density distribution [A/mm2]",
                fontsize=10,
                ha="center",
                pad=20,
            )

            if config["bars"]:
                for b, bar in enumerate(bars_data):
                    fontsize = 10
                    text_shift_y = -1 * fontsize
                    if b %  2:
                        text_shift_y = fontsize

                    if config["bardetails"]:
                        text_line = f"[{b:>2}] {csdm.getComplexModule(bar.current):.1f}A\n dP: {bar.power:.1f}W"
                        text_line_thermal = f"[{bar.dT:.1f}K]"
                    else:
                        text_line = f"[{b:>2}]"
                        text_line_thermal = f"[{bar.dT:.1f}K]"

                    ax.text(
                        (-(len(text_line)//2)*fontsize/2+bar.center[0]) / dXmm,
                        (text_shift_y + bar.center[1]) / dYmm,
                        text_line,
                        fontsize=fontsize,
                        color="black",
                    )
                    bx.text(
                        (-(len(text_line)//2)*fontsize/2+bar.center[0]) / dXmm,
                        (text_shift_y + bar.center[1])/ dYmm,
                        text_line_thermal,
                        fontsize=fontsize,
                        color="black",
                    )
            plt.show()



if __name__ == "__main__":
    if '--cli' in sys.argv:
        # remove --cli from argv so argparse doesn't see it
        sys.argv.remove('--cli')
        # Call the CLI runner
        run_cli()
    else:
        # The existing GUI startup code
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

