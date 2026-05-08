import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os.path
import time
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

# ── Dark UI palette ───────────────────────────────────────────────────────────
BG        = "#1e1e1e"   # window / master background
BG_PANEL  = "#252526"   # panel / LabelFrame background
BG_WIDGET = "#2d2d30"   # entry / text widget fill
ACCENT    = "#007acc"   # blue accent (hover / active)
FG        = "#d4d4d4"   # primary text
FG2       = "#9d9d9d"   # caption / secondary text
C_BG      = "#2a2a2a"   # canvas background
C_BG2     = "#2f2f2f"   # drawing-area fill (paper)
GRID      = "#3c3c3c"   # minor grid lines
GRID_MAJ  = "#525252"   # major grid lines (every 5 cells)
PH_A      = "#c0392b"   # phase-A button colour
PH_B      = "#27ae60"   # phase-B button colour
PH_C      = "#2980b9"   # phase-C button colour
PH_X      = "#4a4a4a"   # erase / neutral button colour
DRAW_A    = "#e74c3c"   # phase-A cell colour on canvas
DRAW_B    = "#2ecc71"   # phase-B cell colour on canvas
DRAW_C    = "#3498db"   # phase-C cell colour on canvas
# ─────────────────────────────────────────────────────────────────────────────


def showXsecArray(event):
    print(XSecArray)


def saveArrayToFile():
    filename = filedialog.asksaveasfilename()
    filename = os.path.normpath(filename)
    if filename:
        saveTheData(filename)


def saveTheData(filename):
    S = csd.cointainer(XSecArray, dXmm, dYmm)
    S.save(filename)
    del S


def loadArrayFromFile():
    filename = filedialog.askopenfilename(filetypes=[("CSD files", "*.csd")])
    filename = os.path.normpath(filename)

    if os.path.isfile(filename) and np.sum(XSecArray) != 0:
        q = messagebox.askquestion(
            "Delete", "This will delete current shape. Are You Sure?", icon="warning"
        )
        if q == "yes":
            loadTheData(filename)
    else:
        if os.path.isfile(filename):
            loadTheData(filename)


def importArrayFromPicture():
    filename = filedialog.askopenfilename(
        filetypes=[("Pictures", "*.jpg *.jpeg *.JPG *.PNG *.png")]
    )
    filename = os.path.normpath(filename)

    if os.path.isfile(filename) and np.sum(XSecArray) != 0:
        q = messagebox.askquestion(
            "Delete", "This will delete current shape. Are You Sure?", icon="warning"
        )
        if q == "yes":
            importTheData(filename)
    else:
        if os.path.isfile(filename):
            importTheData(filename)


def loadTheData(filename):
    global XSecArray, dXmm, dYmm

    print("reading from file :" + filename)
    S = csd.loadObj(filename)
    XSecArray, dXmm, dYmm = S.restore()
    print("dX:{} dY:{}".format(dXmm, dYmm))
    myEntryDx.delete(0, END)
    myEntryDx.insert(END, str(dXmm))
    setParameters()
    printTheArray(XSecArray, canvas=w)
    del S


def importTheData(filename):
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
    print(XSecArray)
    print(str(dXmm) + "[mm] :" + str(dYmm) + "[mm]")
    printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)
    drawGeometryArray(XSecArray)


def clearArrayAndDisplay():
    global XSecArray, dX, dY
    if np.sum(XSecArray) != 0:
        q = messagebox.askquestion(
            "Delete", "This will delete current shape. Are You Sure?", icon="warning"
        )
        if q == "yes":
            XSecArray = np.zeros(XSecArray.shape)
            mainSetup()
            printTheArray(XSecArray, w)
            myEntryDx.delete(0, END)
            myEntryDx.insert(END, str(dXmm))
            setParameters()
    else:
        XSecArray = np.zeros(XSecArray.shape)
        mainSetup()
        printTheArray(XSecArray, w)
        myEntryDx.delete(0, END)
        myEntryDx.insert(END, str(dXmm))
        setParameters()


def subdivideArray():
    global XSecArray, dXmm, dYmm, selectionArray

    if dXmm > 1 and dYmm > 1:
        XSecArray = csd.n_arraySlicer(inputArray=XSecArray, subDivisions=2)

        dXmm = dXmm / 2
        dYmm = dYmm / 2

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


def simplifyArray():
    global XSecArray, dXmm, dYmm, selectionArray

    if dXmm < 30 and dYmm < 30:
        XSecArray = XSecArray[::2, ::2]

        dXmm = dXmm * 2
        dYmm = dYmm * 2

        selectionArray = None
        printTheArray(dataArray=XSecArray, canvas=w)
    else:
        print("No further simplification make sense :)")

    myEntryDx.delete(0, END)
    myEntryDx.insert(END, str(dXmm))
    setParameters()


def extendArray():
    global XSecArray

    rows = XSecArray.shape[0]
    cols = XSecArray.shape[1]

    extension = 10

    extendArray = np.zeros((rows + 2 * extension, cols + 2 * extension))
    extendArray[extension : extension + rows, extension : extension + cols] = XSecArray
    XSecArray = extendArray
    zoomOut()


def showMeForces(*arg):
    global forceCalc

    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Forces calculator")
        forceCalc = gui.forceWindow(root, XSecArray, dXmm, dYmm)


def showMePower(*arg):
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Power Losses Calculator")
        powerCalc = gui.currentDensityWindow(root, XSecArray, dXmm, dYmm)


def showMePro(*arg):
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Pro Power Losses Solver")
        powerCalc = gui.currentDensityWindowPro(root, XSecArray, dXmm, dYmm)


def showMeZ(*arg):
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Impedances Calculator")
        zCalc = gui.zWindow(root, XSecArray, dXmm, dYmm)


def showMeZ3f(*arg):
    setParameters()

    if np.sum(XSecArray) > 0:
        root = Tk()
        root.title("Impedances Calculator")
        zCalc = gui.zWindow3f(root, XSecArray, dXmm, dYmm)


def showReplacer(*arg):
    global XSecArray

    root = Tk()
    root.title("Impedances Calculator")
    TestWindow = gui.geometryModWindow(root, w)

    try:
        sourcePhase = int(input("Source phase [1,2,3]: "))
        toPhase = int(input("to phase [1,2,3]: "))
    except:
        sourcePhase = 0
        toPhase = 0

    if sourcePhase in [1, 2, 3] and toPhase in [1, 2, 3] and sourcePhase != toPhase:
        XSecArray[XSecArray == sourcePhase] = toPhase
        print("Phase {} changed to phase {}".format(sourcePhase, toPhase))
        printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)
    else:
        print("cannot swap!")


def showMeGeom(*arg):
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
    global \
        elementsVector, \
        resultsArray, \
        resultsCurrentVector, \
        frequency, \
        powerLosses, \
        powPh, \
        resultsArrayPower, \
        powerLossesVector

    setParameters()

    if np.sum(XSecArray) > 0:

        config = {
            'frequency': frequency, 'temperature': temperature, 'length': 1000,
            'htc': 5, 'conductivity': 56e6, 'temRcoeff': 3.9e-3, 'material': -1,
            'simple': False, 'markdown': False, 'csv': False, 'verbose': False,
            'draw': False, 'results': True, 'bars': False, 'bardetails': False,
            'geometry': '', 'current': curentRMS,
        }

        verbose = config["verbose"]
        csd.verbose = verbose

        list_of_phases = np.unique(XSecArray).astype(int)
        list_of_phases = [int(n) for n in list_of_phases if n != 0]

        normalized_XsecArr = np.zeros(XSecArray.shape)
        for index, phase_val in enumerate(list_of_phases, start=1):
            normalized_XsecArr[XSecArray == phase_val] = index
        XSecArray = normalized_XsecArr

        number_of_phases = len(list_of_phases)

        Irms = config["current"]
        Icurrent = []
        phi = [120, 0, 240, 120, 0, 240]
        direction = [0, 0, 0, 180, 180, 180]
        x = 0
        for n in range(number_of_phases):
            Icurrent.append((Irms, phi[x] + direction[x]))
            x += 1
            if x >= len(phi):
                x = 0

        f = config["frequency"]
        length = config["length"]
        t = config["temperature"]

        M_list = csdos.read_file_to_list("setup/materials.txt")[1:]
        MaterialsDB = csdos.get_material_from_list(M_list) if M_list else []

        if MaterialsDB:
            this_material = MaterialsDB[0]
        else:
            this_material = csdos.Material(
                "Cu", config["conductivity"], config["temRcoeff"], 0, 0
            )
        phases_material = [this_material] * number_of_phases

        (
            resultsCurrentVector,
            powerResults,
            elementsVector,
            powerLossesSolution,
            complexCurrent,
            vPh,
            mi_r_weighted,
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

        showResults()


def drawGeometryArray(theArrayToDisplay):
    global figGeom, geomax, geomim

    title_font = {"size": "11", "color": "black", "weight": "normal"}
    axis_font = {"size": "10"}

    my_cmap = matplotlib.cm.get_cmap("jet")
    my_cmap.set_under("w")

    figGeom = plt.figure(1)

    vmin = 0 if np.sum(theArrayToDisplay) == 0 else 0.8

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
        min_row = int(np.min(elementsVector[:, 0]))
        max_row = int(np.max(elementsVector[:, 0]) + 1)

        min_col = int(np.min(elementsVector[:, 1]))
        max_col = int(np.max(elementsVector[:, 1]) + 1)

        resultsArrayDisplay = resultsArray[min_row:max_row, min_col:max_col]
        resultsArrayDisplay2 = resultsArrayPower[min_row:max_row, min_col:max_col]

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
            alpha=0.9,
            fraction=0.046,
        )
        plt.axis("scaled")

        phase_line = " ".join(f"ph{i}: {dP:.2f}[W]" for i, dP in enumerate(powPh))
        ax.set_title(
            f"{frequency}[Hz] / {curentRMS}[A] / {temperature}[°C]\n"
            f"Power Losses {powerLosses:.2f}[W]\n{phase_line}",
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
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, -1, 0, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def shiftR():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 1, 0, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def shiftU():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 0, -1, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def shiftD():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 0, 1, XSecArray)
    print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
    printTheArray(XSecArray, canvas=w)


def mainSetup(startSize=3):
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

ic_currents  = []   # populated by getCanvas / InterCode from .ic current() lines
ic_materials = []   # populated by getCanvas / InterCode from .ic material() lines


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
    global canvasElements, selectShadowBox, selectEndPoint, selectStartPoint

    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

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
            pass
    canvasElements = []

    # Drawing-area background
    canvasElements.append(
        canvas.create_rectangle(
            startX, startY,
            canvasWidth - startX, canvasHeight - startY,
            fill=C_BG2, outline=GRID,
        )
    )

    colorList = [DRAW_A, DRAW_B, DRAW_C]

    for Row in range(elementsInY):
        for Col in range(elementsInX):
            theNumber = int(dataArray[Row][Col])
            if theNumber in [1, 2, 3]:
                canvasElements.append(
                    canvas.create_rectangle(
                        startX + Col * dXY,
                        startY + Row * dXY,
                        startX + Col * dXY + dXY,
                        startY + Row * dXY + dXY,
                        fill=colorList[theNumber - 1],
                        outline="",
                    )
                )

            if Row == elementsInY - 1:
                lineFillColor = GRID
                lineWidth = 1

                if (Col + globalX) % 5 == 0 and lineSkip == 1:
                    lineFillColor = GRID_MAJ
                    lineWidth = 2

                if Col % lineSkip == 0:
                    canvasElements.append(
                        canvas.create_line(
                            startX + Col * dXY, startY,
                            startX + Col * dXY, canvasHeight - startY,
                            fill=lineFillColor, width=lineWidth,
                        )
                    )

        lineFillColor = GRID
        lineWidth = 1
        if (Row + globalY) % 5 == 0 and lineSkip == 1:
            lineFillColor = GRID_MAJ
            lineWidth = 2

        if Row % lineSkip == 0:
            canvasElements.append(
                canvas.create_line(
                    startX, startY + Row * dXY,
                    canvasWidth - startX, startY + Row * dXY,
                    fill=lineFillColor, width=lineWidth,
                )
            )

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
            startX + C1 * dXY,
            startY + R1 * dXY,
            startX + C2 * dXY,
            startY + R2 * dXY,
            fill="", outline="#ffd700", width=3,
        )


def setPoint(event):
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

    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def resetPoint(event):
    setUpPoint(
        event,
        Set=0,
        dataArray=zoomInArray(XSecArray, globalZoom, globalX, globalY),
        canvas=w,
    )

    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def setUpPoint(event, Set, dataArray, canvas):
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
    global shadowBox
    try:
        canvas.delete(shadowBox)
    except:
        pass

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
            startX + Col * dXY,
            startY + Row * dXY,
            startX + Col * dXY + dXY,
            startY + Row * dXY + dXY,
            fill="#3a3a3a", outline="#ffd700",
        )


def shadowPoint(event):
    moveShadowPoint(event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)


def startSelection(event, dataArray, canvas):
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
                startX + C1 * dXY,
                startY + R1 * dXY,
                startX + C2 * dXY,
                startY + R2 * dXY,
                fill="", outline="#ffd700", width=3,
            )


def endSelection(event):
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

        selectionMaskArray = np.empty_like(
            zoomInArray(XSecArray, globalZoom, globalX, globalY)
        )
        selectionMaskArray[R1:R2, C1:C2] = 1

        selectionArray = np.copy(
            zoomInArray(XSecArray, globalZoom, globalX, globalY)[R1:R2, C1:C2]
        )

        phase.set(5)


def pasteSelectionAtPoint(event, dataArray, canvas):
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
            R2 = min(R1 + pasteRows, XSecArray.shape[0])

            C1 = Col + globalX
            C2 = min(C1 + pasteCols, XSecArray.shape[1])

            selectedPasteMode = paste_mode.get()

            if selectedPasteMode == 1:
                XSecArray[R1:R2, C1:C2] = np.where(
                    selectionArray[: R2 - R1, : C2 - C1] > 0,
                    selectionArray[: R2 - R1, : C2 - C1],
                    XSecArray[R1:R2, C1:C2],
                )

            elif selectedPasteMode == 2:
                XSecArray[R1:R2, C1:C2] = selectionArray[: R2 - R1, : C2 - C1]

            elif 10 < selectedPasteMode < 20:
                targetPhase = selectedPasteMode - 10
                XSecArray[R1:R2, C1:C2] = np.where(
                    selectionArray[: R2 - R1, : C2 - C1] > 0,
                    targetPhase,
                    XSecArray[R1:R2, C1:C2],
                )

            printTheArray(dataArray, canvas)


def InterCode():
    global ic_currents, ic_materials
    codeLines = text_input.get("1.0", END).split("\n")
    codeSteps, ic_currents, ic_materials = ic.textToCode(codeLines)

    if codeSteps:
        for step in codeSteps:
            step[0](*step[1], XSecArray=XSecArray, dXmm=dXmm)

    redraw()


def getCanvas():
    global ic_currents, ic_materials
    codeLines = text_input.get("1.0", END).split("\n")
    codeSteps, ic_currents, ic_materials = ic.textToCode(codeLines)

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
if __name__ == "__main__":
    master = Tk()
    master.title("Cross Section Designer")
    master.configure(bg=BG)

    img = PhotoImage(file="CSDico.gif")
    master.tk.call("wm", "iconphoto", master._w, img)

    # ── Apply dark ttk theme ──────────────────────────────────────────────────
    style = ttk.Style(master)
    style.theme_use("clam")
    style.configure(".",
        background=BG_PANEL,
        foreground=FG,
        font=("Helvetica", 10),
        troughcolor=BG_WIDGET,
        selectbackground=ACCENT,
        selectforeground="white",
    )
    style.configure("TFrame", background=BG_PANEL)
    style.configure("TLabelframe",
        background=BG_PANEL,
        bordercolor="#3c3c3c",
        relief="groove",
    )
    style.configure("TLabelframe.Label",
        background=BG_PANEL,
        foreground=FG2,
        font=("Helvetica", 9, "bold"),
    )
    style.configure("TLabel", background=BG_PANEL, foreground=FG)
    style.configure("TButton",
        background=BG_WIDGET, foreground=FG,
        relief="flat", borderwidth=1, padding=4,
    )
    style.map("TButton",
        background=[("active", ACCENT), ("pressed", ACCENT)],
        foreground=[("active", "white"), ("pressed", "white")],
    )
    style.configure("TEntry",
        fieldbackground=BG_WIDGET, foreground=FG, insertcolor=FG,
        bordercolor="#3c3c3c", lightcolor=BG_WIDGET, darkcolor=BG_WIDGET,
    )
    style.configure("Status.TFrame", background=BG_WIDGET)
    style.configure("Status.TLabel",
        background=BG_WIDGET, foreground=FG2,
        font=("Helvetica", 9),
    )
    # ─────────────────────────────────────────────────────────────────────────

    canvas_width = 650
    canvas_height = 650

    master.resizable(width=True, height=True)

    mainSetup()

    master.bind("<Configure>", redraw)

    w = Canvas(master, width=canvas_width, height=canvas_height)
    w.configure(background=C_BG)
    w.grid(row=1, column=1, columnspan=5, rowspan=25, sticky=W + E + N + S, padx=1, pady=1)

    canvasElements = []
    shadowBox = None

    inSelectMode = False
    selectStartPoint = None
    selectEndPoint = None
    selectShadowBox = None

    selectionMaskArray = None
    selectionArray = None

    # ── Menu bar ──────────────────────────────────────────────────────────────
    menu_bar = Menu(master,
        bg=BG_PANEL, fg=FG,
        activebackground=ACCENT, activeforeground="white",
        relief="flat", borderwidth=0,
    )

    file_menu = Menu(menu_bar, tearoff=0,
        bg=BG_PANEL, fg=FG,
        activebackground=ACCENT, activeforeground="white",
    )
    file_menu.add_command(label="New geometry", command=clearArrayAndDisplay)
    file_menu.add_separator()
    file_menu.add_command(label="Load from file", command=loadArrayFromFile)
    file_menu.add_command(label="Import from picture", command=importArrayFromPicture)
    file_menu.add_separator()
    file_menu.add_command(label="Save to file", command=saveArrayToFile)
    menu_bar.add_cascade(label="File", menu=file_menu)

    analyze_menu = Menu(menu_bar, tearoff=0,
        bg=BG_PANEL, fg=FG,
        activebackground=ACCENT, activeforeground="white",
    )
    analyze_menu.add_command(label="Power Losses ProSolver", command=showMePro)
    analyze_menu.add_command(label="Electro Dynamic Forces", command=showMeForces)
    analyze_menu.add_command(label="Equivalent Impedance Model", command=showMeZ)
    analyze_menu.add_command(label="Equivalent Impedance Model 3f Shunt", command=showMeZ3f)
    menu_bar.add_cascade(label="Analyze...", menu=analyze_menu)

    geometry_menu = Menu(menu_bar, tearoff=0,
        bg=BG_PANEL, fg=FG,
        activebackground=ACCENT, activeforeground="white",
    )
    geometry_menu.add_command(label="Pattern", command=showMeGeom)
    geometry_menu.add_command(label="Swap", command=showReplacer)
    geometry_menu.add_separator()
    geometry_menu.add_command(label="Subdivide(+)", command=subdivideArray)
    geometry_menu.add_command(label="Simplify(-)", command=simplifyArray)
    geometry_menu.add_separator()
    geometry_menu.add_command(label="Extend Canvas", command=extendArray)
    menu_bar.add_cascade(label="Geometry", menu=geometry_menu)

    view_menu = Menu(menu_bar, tearoff=0,
        bg=BG_PANEL, fg=FG,
        activebackground=ACCENT, activeforeground="white",
    )
    view_menu.add_command(label="Open CAD view window", command=displayArrayAsImage)
    menu_bar.add_cascade(label="View", menu=view_menu)

    master.config(menu=menu_bar)

    # ── Tool selector panel ───────────────────────────────────────────────────
    A_icon_white      = PhotoImage(file="csdicons/A_white.png")
    B_icon_white      = PhotoImage(file="csdicons/B_white.png")
    C_icon_white      = PhotoImage(file="csdicons/C_white.png")
    cut_icon_white    = PhotoImage(file="csdicons/cut_white.png")
    select_icon_white = PhotoImage(file="csdicons/select_white.png")
    paste_icon_white  = PhotoImage(file="csdicons/paste_white.png")
    paste_icon_all    = PhotoImage(file="csdicons/paste_full.png")
    paste_icon_A      = PhotoImage(file="csdicons/paste_A.png")
    paste_icon_B      = PhotoImage(file="csdicons/paste_B.png")
    paste_icon_C      = PhotoImage(file="csdicons/paste_C.png")

    phase_frame = ttk.LabelFrame(master, text="Active operation")
    phase_frame.grid(row=1, column=8, columnspan=3, padx=4, pady=4)

    phase = IntVar()
    phase.set(1)

    def _phase_btn(parent, image, value, bg_color):
        return Radiobutton(
            parent, image=image, variable=phase, value=value,
            indicatoron=0, height=32, width=32,
            bg=bg_color, activebackground=bg_color,
            selectcolor=bg_color, highlightbackground=bg_color,
            relief="flat", cursor="hand2",
        )

    _phase_btn(phase_frame, A_icon_white,      1, PH_A).grid(row=0, column=0, padx=2, pady=2)
    _phase_btn(phase_frame, B_icon_white,      2, PH_B).grid(row=0, column=1, padx=2, pady=2)
    _phase_btn(phase_frame, C_icon_white,      3, PH_C).grid(row=0, column=2, padx=2, pady=2)
    _phase_btn(phase_frame, cut_icon_white,    0, PH_X).grid(row=1, column=0, padx=2, pady=2)
    _phase_btn(phase_frame, select_icon_white, 4, PH_X).grid(row=1, column=1, padx=2, pady=2)
    _phase_btn(phase_frame, paste_icon_white,  5, PH_X).grid(row=1, column=2, padx=2, pady=2)

    # ── Paste mode panel ──────────────────────────────────────────────────────
    paste_frame = ttk.LabelFrame(master, text="Paste mode")
    paste_frame.grid(row=3, column=8, columnspan=3, padx=4, pady=4)

    paste_mode = IntVar()
    paste_mode.set(1)

    def _paste_btn(parent, image, value, bg_color=PH_X):
        return Radiobutton(
            parent, image=image, variable=paste_mode, value=value,
            indicatoron=0, height=32, width=32,
            bg=bg_color, activebackground=ACCENT,
            selectcolor=ACCENT, highlightbackground=bg_color,
            relief="flat", cursor="hand2",
        )

    _paste_btn(paste_frame, paste_icon_white, 1      ).grid(row=0, column=0, padx=2, pady=2)
    _paste_btn(paste_frame, paste_icon_all,   2      ).grid(row=0, column=1, padx=2, pady=2)
    _paste_btn(paste_frame, paste_icon_A,    11, PH_A).grid(row=1, column=0, padx=2, pady=2)
    _paste_btn(paste_frame, paste_icon_B,    12, PH_B).grid(row=1, column=1, padx=2, pady=2)
    _paste_btn(paste_frame, paste_icon_C,    13, PH_C).grid(row=1, column=2, padx=2, pady=2)

    # ── Shift selected-phase panel ────────────────────────────────────────────
    up_icon_white    = PhotoImage(file="csdicons/up_white.png")
    down_icon_white  = PhotoImage(file="csdicons/down_white.png")
    left_icon_white  = PhotoImage(file="csdicons/left_white.png")
    right_icon_white = PhotoImage(file="csdicons/right_white.png")

    mod_frame = ttk.LabelFrame(master, text="Shift sel. phase")
    mod_frame.grid(row=6, column=8, columnspan=3, sticky="S", padx=4, pady=4)

    def _shift_btn(parent, image, cmd):
        return Button(
            parent, image=image, width=24, height=24,
            bg=BG_PANEL, activebackground=ACCENT,
            relief="flat", cursor="hand2",
            command=cmd, repeatdelay=100, repeatinterval=100,
        )

    _shift_btn(mod_frame, left_icon_white,  shiftL).grid(row=1, column=0, padx=5, pady=5)
    _shift_btn(mod_frame, right_icon_white, shiftR).grid(row=1, column=2, padx=5, pady=5)
    _shift_btn(mod_frame, up_icon_white,    shiftU).grid(row=0, column=1, padx=5, pady=5)
    _shift_btn(mod_frame, down_icon_white,  shiftD).grid(row=1, column=1, padx=5, pady=5)

    # ── Grid-size entry row ───────────────────────────────────────────────────
    Label(master, text="", height=3, bg=BG).grid(row=5, column=0)

    analysisDX = Label(
        master, text="dx:\n " + str(dXmm) + "[mm]",
        height=2, bg=BG, fg=FG, font=("Helvetica", 9),
    )
    analysisDX.grid(row=5, column=8, columnspan=1)

    myEntryDx = ttk.Entry(master, width=5)
    myEntryDx.insert(END, str(dXmm))
    myEntryDx.grid(row=5, column=9, columnspan=1, padx=1, pady=1)
    myEntryDx.bind("<Return>", setParameters)
    myEntryDx.bind("<FocusOut>", setParameters)

    analysisDY = Label(
        master, text="dy:\n " + str(dYmm) + "[mm]",
        height=2, bg=BG, fg=FG, font=("Helvetica", 9),
    )
    analysisDY.grid(row=5, column=10, columnspan=1)

    # ── View navigation panel ─────────────────────────────────────────────────
    zoom_in_icon  = PhotoImage(file="csdicons/zoomin.png")
    zoom_out_icon = PhotoImage(file="csdicons/zoomout.png")
    up_icon       = PhotoImage(file="csdicons/up.png")
    down_icon     = PhotoImage(file="csdicons/down.png")
    left_icon     = PhotoImage(file="csdicons/left.png")
    right_icon    = PhotoImage(file="csdicons/right.png")

    navi_frame = ttk.LabelFrame(master, text="View navi")
    navi_frame.grid(row=21, column=8, columnspan=3, sticky="S", padx=4, pady=4)

    def _navi_btn(parent, image, cmd, bw=32, bh=32):
        return Button(
            parent, image=image, width=bw, height=bh,
            bg=BG_PANEL, activebackground=ACCENT,
            relief="flat", cursor="hand2",
            command=cmd, repeatdelay=100, repeatinterval=100,
        )

    _navi_btn(navi_frame, zoom_in_icon,  zoomIn        ).grid(row=0, column=0, padx=5, pady=5)
    _navi_btn(navi_frame, zoom_out_icon, zoomOut       ).grid(row=0, column=2, padx=5, pady=5)
    _navi_btn(navi_frame, up_icon,   zoomU, 24, 24).grid(row=0, column=1, padx=5, pady=5)
    _navi_btn(navi_frame, left_icon,  zoomL, 24, 24).grid(row=1, column=0, padx=5, pady=5)
    _navi_btn(navi_frame, right_icon, zoomR, 24, 24).grid(row=1, column=2, padx=5, pady=5)
    _navi_btn(navi_frame, down_icon,  zoomD, 24, 24).grid(row=1, column=1, padx=5, pady=5)

    # ── Inner-code panel ──────────────────────────────────────────────────────
    code_frame = ttk.LabelFrame(master, text="inner-code window")
    code_frame.grid(row=1, column=11, rowspan=25, columnspan=3, sticky="N", padx=5, pady=0)

    text_input = Text(
        code_frame, height=20, width=25,
        bg=BG_WIDGET, fg=FG, insertbackground=FG,
        font=("Courier New", 10), relief="flat", borderwidth=0,
        selectbackground=ACCENT, selectforeground="white",
        highlightbackground="#3c3c3c", highlightthickness=1,
    )
    text_input.grid(row=1, column=1, columnspan=3, padx=8, pady=8)

    ttk.Button(code_frame, text="Execute InnerCode as is", command=InterCode).grid(
        row=19, column=1, columnspan=3, padx=6, pady=3, sticky="EW",
    )
    ttk.Button(code_frame, text="Create new geometry", command=getCanvas).grid(
        row=20, column=1, columnspan=3, padx=6, pady=3, sticky="EW",
    )

    # ── Canvas bindings ───────────────────────────────────────────────────────
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

    # ── Status bar ────────────────────────────────────────────────────────────
    status_frame = ttk.Frame(master, style="Status.TFrame", height=26)
    status_frame.grid(row=27, column=0, columnspan=15, sticky="EW")
    status_frame.grid_propagate(False)

    ttk.Label(
        status_frame,
        text="  LMB: draw  •  RMB: erase  •  Arrow keys: pan  •  Analyze menu: run solver",
        style="Status.TLabel",
    ).grid(row=0, column=0, sticky="W", padx=8, pady=4)

    # ── Grid resize weights ───────────────────────────────────────────────────
    master.grid_rowconfigure(0, weight=0)
    master.grid_rowconfigure(12, weight=1)
    master.grid_columnconfigure(0, weight=0)
    master.grid_columnconfigure(1, weight=1)

    master.update()

    canvas_height = w.winfo_height()
    canvas_width = w.winfo_width()

    printTheArray(dataArray=XSecArray, canvas=w)

    mainloop()
