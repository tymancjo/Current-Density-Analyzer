import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tkinter import *
from tkinter import filedialog, messagebox
import customtkinter as ctk
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

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Light UI palette ──────────────────────────────────────────────────────────
BG        = "#f0f2f5"   # window background
BG_PANEL  = "#ffffff"   # panel / card background
BG_WIDGET = "#f5f7fa"   # entry / textbox fill
TOOLBAR   = "#e4e7ec"   # top toolbar background
SEP       = "#d1d5db"   # separator / border
ACCENT    = "#0078d4"   # primary accent
ACCENT_H  = "#106ebe"   # accent hover
FG        = "#111827"   # primary text
FG2       = "#6b7280"   # secondary / caption text
CANVAS_BG = "#f8f9fa"   # drawing canvas background
CANVAS_P  = "#ffffff"   # drawing-area fill ("paper")
GRID      = "#e5e7eb"   # minor grid lines
GRID_MAJ  = "#c8ccd0"   # major grid lines
PH_A      = "#c0392b"   # phase-A button
PH_B      = "#27ae60"   # phase-B button
PH_C      = "#2980b9"   # phase-C button
PH_X      = "#8e9aaf"   # erase / neutral button
DRAW_A    = "#e74c3c"   # phase-A canvas cells
DRAW_B    = "#2ecc71"   # phase-B canvas cells
DRAW_C    = "#3498db"   # phase-C canvas cells
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
    myEntryDx.delete(0, "end")
    myEntryDx.insert(0, str(dXmm))
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

        myEntryDx.delete(0, "end")
        myEntryDx.insert(0, str(dXmm))

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
            myEntryDx.delete(0, "end")
            myEntryDx.insert(0, str(dXmm))
            setParameters()
    else:
        XSecArray = np.zeros(XSecArray.shape)
        mainSetup()
        printTheArray(XSecArray, w)
        myEntryDx.delete(0, "end")
        myEntryDx.insert(0, str(dXmm))
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

    myEntryDx.delete(0, "end")
    myEntryDx.insert(0, str(dXmm))
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

    myEntryDx.delete(0, "end")
    myEntryDx.insert(0, str(dXmm))
    setParameters()


def extendArray():
    global XSecArray

    rows = XSecArray.shape[0]
    cols = XSecArray.shape[1]
    extension = 10

    ext = np.zeros((rows + 2 * extension, cols + 2 * extension))
    ext[extension : extension + rows, extension : extension + cols] = XSecArray
    XSecArray = ext
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
        powerCalc = gui.currentDensityWindowPro(
            root, XSecArray, dXmm, dYmm,
            ic_currents=ic_currents,
            ic_materials=ic_materials,
            ic_custom_materials=ic_custom_materials,
        )


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
    root.title("Geometry modifier")
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
        powerLossesVector, \
        temperatureArray

    setParameters()

    if np.sum(XSecArray) > 0:

        config = {
            'frequency': frequency, 'temperature': temperature, 'length': 1000,
            'htc': 5, 'conductivity': 56e6, 'temRcoeff': 3.9e-3, 'material': -1,
            'simple': False, 'markdown': False, 'csv': False, 'verbose': False,
            'draw': False, 'results': True, 'bars': True, 'bardetails': False,
            'geometry': '', 'current': curentRMS,
        }

        verbose = config["verbose"]
        csd.verbose = verbose

        list_of_phases = np.unique(XSecArray).astype(int)
        list_of_phases = [int(n) for n in list_of_phases if n != 0]
        original_phase_index = {index: phase for index, phase in enumerate(list_of_phases)}

        normalized_XsecArr = np.zeros(XSecArray.shape)
        for index, phase_val in enumerate(list_of_phases, start=1):
            normalized_XsecArr[XSecArray == phase_val] = index
        XSecArray = normalized_XsecArr

        number_of_phases = len(list_of_phases)

        # Build new_phase_index: maps original .ic phase ID → 0-based solver index
        new_phase_index = {phase_val: idx for idx, phase_val in enumerate(list_of_phases)}

        Irms = config["current"]
        Icurrent = []
        phi = [120, 0, 240, 120, 0, 240]
        direction = [0, 0, 0, 180, 180, 180]
        x = 0
        for n in range(number_of_phases):
            Icurrent.append([Irms, phi[x] + direction[x]])
            x += 1
            if x >= len(phi):
                x = 0

        if ic_currents:
            for entry in ic_currents:
                phase_id = int(entry[0])
                if phase_id in new_phase_index:
                    idx = new_phase_index[phase_id]
                    Icurrent[idx] = [float(entry[1]), float(entry[2]) + float(entry[3])]

        f = config["frequency"]
        length = config["length"]
        t = config["temperature"]
        HTC = config["htc"]

        M_list = csdos.read_file_to_list("setup/materials.txt")[1:]
        MaterialsDB = csdos.get_material_from_list(M_list) if M_list else []

        default_material = MaterialsDB[0] if MaterialsDB else csdos.Material(
            "Cu", config["conductivity"], config["temRcoeff"], 0, 0
        )

        # Build a phase_id → Material mapping from ic_materials + ic_custom_materials.
        # Entries where the material ref is an int use the library by index;
        # entries where it is a string use a custom material defined via defmat().
        phase_material_map = {}
        for entry in ic_materials:
            phase_id = int(entry[0])
            mat_ref  = entry[1]
            if isinstance(mat_ref, str):
                if mat_ref in ic_custom_materials:
                    m = ic_custom_materials[mat_ref]
                    phase_material_map[phase_id] = csdos.Material(
                        mat_ref,
                        m['sigma'], m['alpha'],
                        m['rho'],   m['cp'],
                        m['mi_r'],  m['k'] if m['k'] else 400,
                    )
                else:
                    print(f"[vectorize] Unknown material name '{mat_ref}' — using default")
            else:
                idx = int(mat_ref)
                if 0 <= idx < len(MaterialsDB):
                    phase_material_map[phase_id] = MaterialsDB[idx]
                else:
                    print(f"[vectorize] Material index {idx} out of range — using default")

        phases_material = [
            phase_material_map.get(int(phase_val), default_material)
            for phase_val in list_of_phases
        ]

        (
            resultsCurrentVector,
            powerResults,
            elementsVector,
            powerLossesSolution,
            complexCurrent,
            vPh,
            mi_r_weighted,
            _phase_U,
            _phase_I,
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

        # ── Thermal analysis ──────────────────────────────────────────────────
        temperatureArray = None
        try:
            currentsDraw = csdm.recreateresultsArray(
                elementsVector, complexCurrent, XSecArray, dtype=complex
            )
            powerDraw = csdm.recreateresultsArray(
                elementsVector, powerLossesSolution, XSecArray
            )

            conductorsXsecArr, total_conductors, phases_conductors = csdf.getConductors(
                XSecArray, vPh
            )

            bars_data = []
            for b in range(1, total_conductors + 1):
                bar = csdf.the_bar()
                bar.elements = csdm.arrayVectorize(
                    conductorsXsecArr, phaseNumber=b, dXmm=dXmm, dYmm=dYmm
                )
                bars_data.append(bar)

            for i, ph_bars in enumerate(phases_conductors):
                for b in ph_bars:
                    bars_data[b - 1].phase = i
                    bars_data[b - 1].material = phases_material[i]

            for i, bar in enumerate(bars_data):
                bar.number = i
                bar.perymiter = csdf.getPerymiter(bar.elements, XSecArray, dXmm, dYmm)

                x_sum = y_sum = 0.0
                for element in bar.elements:
                    R = int(element[0])
                    C = int(element[1])
                    bar.current += currentsDraw[R, C]
                    bar.power += powerDraw[R, C]
                    x_sum += element[2]
                    y_sum += element[3]

                bar.center = [x_sum / len(bar.elements), y_sum / len(bar.elements)]
                bar.length = length
                bar.xsection = len(bar.elements) * dXmm * dYmm
                bar.Rth = (
                    bar.length * 1e-3
                    / (bar.xsection * 1e-6 * bar.material.thermal_conductivity)
                )
                bar.R = (
                    bar.length * 1e-3
                    / (bar.xsection * 1e-6 * bar.material.sigma)
                )

            for bar in bars_data:
                bar.phase = original_phase_index[bar.phase]

            csds.solve_thermal_for_bars(bars_data, HTC=HTC)
            temperatureArray = csdf.recreate_temperature_array(bars_data, XSecArray.shape)

            print(f"Thermal analysis done — {total_conductors} bars detected")
            for bar in bars_data:
                print(f"  Bar {bar.number} (ph{bar.phase})  ΔT={bar.dT:.2f} K")

        except Exception as e:
            print(f"Thermal analysis skipped: {e}")

        showResults()


def drawGeometryArray(theArrayToDisplay):
    global figGeom, geomax, geomim

    title_font = {"size": "11", "color": "black", "weight": "normal"}
    axis_font = {"size": "10"}

    my_cmap = matplotlib.colormaps["jet"]
    my_cmap.set_under("w")

    figGeom = plt.figure(1)
    vmin = 0 if np.sum(theArrayToDisplay) == 0 else 0.8

    geomax = figGeom.add_subplot(1, 1, 1)

    plotWidth = theArrayToDisplay.shape[1] * dXmm
    plotHeight = theArrayToDisplay.shape[0] * dYmm

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

    if np.sum(resultsArray) == 0:
        print("No results available! Run the analysis first.")
        return

    min_row = int(np.min(elementsVector[:, 0]))
    max_row = int(np.max(elementsVector[:, 0]) + 1)
    min_col = int(np.min(elementsVector[:, 1]))
    max_col = int(np.max(elementsVector[:, 1]) + 1)

    cd_display = resultsArray[min_row:max_row, min_col:max_col]
    plotWidth  = cd_display.shape[1] * dXmm
    plotHeight = cd_display.shape[0] * dYmm

    has_temp = temperatureArray is not None and np.sum(temperatureArray) > 0

    if has_temp:
        fig = plt.figure("Results Window", figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
        ax_cd   = fig.add_subplot(gs[0])
        ax_temp = fig.add_subplot(gs[1])
    else:
        fig = plt.figure("Results Window", figsize=(6, 5))
        ax_cd = fig.add_subplot(1, 1, 1)

    # ── Current density plot ──────────────────────────────────────────────────
    my_cmap = matplotlib.colormaps["jet"]
    my_cmap.set_under("w")

    im_cd = ax_cd.imshow(
        cd_display,
        cmap=my_cmap,
        interpolation="none",
        vmin=0.8 * np.min(resultsCurrentVector),
        extent=[0, plotWidth, plotHeight, 0],
    )
    fig.colorbar(im_cd, ax=ax_cd, orientation="vertical",
                 label="Current Density [A/mm²]", fraction=0.046)
    plt.axes(ax_cd)
    plt.axis("scaled")

    phase_line = " ".join(f"ph{i}: {dP:.2f}[W]" for i, dP in enumerate(powPh))
    ax_cd.set_title(
        f"{frequency}[Hz] / {curentRMS}[A] / {temperature}[°C]\n"
        f"Power Losses {powerLosses:.2f}[W]\n{phase_line}",
        **title_font,
    )
    ax_cd.set_xlabel("size [mm]", **axis_font)
    ax_cd.set_ylabel("size [mm]", **axis_font)

    # ── Temperature rise plot ─────────────────────────────────────────────────
    if has_temp:
        temp_display = temperatureArray[min_row:max_row, min_col:max_col]

        temp_cmap = matplotlib.colormaps["YlOrRd"]
        temp_cmap.set_under("w")

        t_max = np.max(temp_display)
        t_min_pos = np.min(temp_display[temp_display > 0]) if np.any(temp_display > 0) else 0.1

        im_t = ax_temp.imshow(
            temp_display,
            cmap=temp_cmap,
            interpolation="none",
            vmin=t_min_pos * 0.8,
            vmax=t_max * 1.05 if t_max > 0 else 1,
            extent=[0, plotWidth, plotHeight, 0],
        )
        fig.colorbar(im_t, ax=ax_temp, orientation="vertical",
                     label="Temperature Rise ΔT [K]", fraction=0.046)
        plt.axes(ax_temp)
        plt.axis("scaled")

        ax_temp.set_title(
            f"Thermal Analysis\nMax ΔT = {t_max:.2f} K",
            **title_font,
        )
        ax_temp.set_xlabel("size [mm]", **axis_font)
        ax_temp.set_ylabel("size [mm]", **axis_font)

    fig.autofmt_xdate(bottom=0.2, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def shiftL():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, -1, 0, XSecArray)
    printTheArray(XSecArray, canvas=w)


def shiftR():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 1, 0, XSecArray)
    printTheArray(XSecArray, canvas=w)


def shiftU():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 0, -1, XSecArray)
    printTheArray(XSecArray, canvas=w)


def shiftD():
    actualPhase = phase.get()
    csd.n_shiftPhase(actualPhase, 0, 1, XSecArray)
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


ic_currents        = []   # populated by getCanvas / InterCode from .ic current() lines
ic_materials       = []   # populated by getCanvas / InterCode from .ic material() lines
ic_custom_materials = {}  # populated from .ic defmat() lines


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

    analysisDX.configure(text=f"dx: {dXmm} mm")
    analysisDY.configure(text=f"dy: {dYmm} mm")


def printTheArray(dataArray, canvas):
    global canvasElements, selectShadowBox, selectEndPoint, selectStartPoint

    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth  = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY
    dXY = min(dX, dY)

    lineSkip = 1
    if dXY <= 2:
        lineSkip = 5
    elif dXY < 5:
        lineSkip = 2

    startX = (canvasWidth  - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    for el in canvasElements:
        try:
            canvas.delete(el)
        except:
            pass
    canvasElements = []

    canvasElements.append(
        canvas.create_rectangle(
            startX, startY,
            canvasWidth - startX, canvasHeight - startY,
            fill=CANVAS_P, outline=GRID,
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
                lc = GRID
                lw = 1
                if (Col + globalX) % 5 == 0 and lineSkip == 1:
                    lc = GRID_MAJ
                    lw = 2
                if Col % lineSkip == 0:
                    canvasElements.append(
                        canvas.create_line(
                            startX + Col * dXY, startY,
                            startX + Col * dXY, canvasHeight - startY,
                            fill=lc, width=lw,
                        )
                    )

        lc = GRID
        lw = 1
        if (Row + globalY) % 5 == 0 and lineSkip == 1:
            lc = GRID_MAJ
            lw = 2
        if Row % lineSkip == 0:
            canvasElements.append(
                canvas.create_line(
                    startX, startY + Row * dXY,
                    canvasWidth - startX, startY + Row * dXY,
                    fill=lc, width=lw,
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
            startX + C1 * dXY, startY + R1 * dXY,
            startX + C2 * dXY, startY + R2 * dXY,
            fill="", outline=ACCENT, width=2,
        )


def setPoint(event):
    actualPhase = phase.get()

    if actualPhase < 4:
        setUpPoint(event, actualPhase, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)
    elif actualPhase == 4:
        startSelection(event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)
    elif actualPhase == 5:
        pasteSelectionAtPoint(event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)

    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def resetPoint(event):
    setUpPoint(event, Set=0, dataArray=zoomInArray(XSecArray, globalZoom, globalX, globalY), canvas=w)

    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def setUpPoint(event, Set, dataArray, canvas):
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth  = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY
    dXY = min(dX, dY)

    startX = (canvasWidth  - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    if (
        startX < event.x < canvasWidth  - startX
        and startY < event.y < canvasHeight - startY
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
    canvasWidth  = canvas.winfo_width()

    dX = canvasWidth / elementsInX
    dY = canvasHeight / elementsInY
    dXY = min(dX, dY)

    startX = (canvasWidth  - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    if (
        startX < event.x < canvasWidth  - startX
        and startY < event.y < canvasHeight - startY
    ):
        Col = int((event.x - startX) / dXY)
        Row = int((event.y - startY) / dXY)

        shadowBox = canvas.create_rectangle(
            startX + Col * dXY,
            startY + Row * dXY,
            startX + Col * dXY + dXY,
            startY + Row * dXY + dXY,
            fill="#dce8f5", outline=ACCENT, width=1,
        )


def shadowPoint(event):
    moveShadowPoint(event, zoomInArray(XSecArray, globalZoom, globalX, globalY), w)


def startSelection(event, dataArray, canvas):
    global inSelectMode, selectStartPoint, selectEndPoint, selectShadowBox

    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    canvasHeight = canvas.winfo_height()
    canvasWidth  = canvas.winfo_width()

    dXY = min(canvasWidth / elementsInX, canvasHeight / elementsInY)
    startX = (canvasWidth  - dXY * elementsInX) / 2
    startY = (canvasHeight - dXY * elementsInY) / 2

    if (
        startX < event.x < canvasWidth  - startX
        and startY < event.y < canvasHeight - startY
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
                startX + C1 * dXY, startY + R1 * dXY,
                startX + C2 * dXY, startY + R2 * dXY,
                fill="", outline=ACCENT, width=2,
            )


def endSelection(event):
    global inSelectMode, selectStartPoint, selectEndPoint, selectionMaskArray, selectionArray

    if inSelectMode and selectEndPoint is not None and selectStartPoint is not None:
        inSelectMode = False

        R1 = min(selectEndPoint[0], selectStartPoint[0])
        R2 = max(selectEndPoint[0], selectStartPoint[0])
        C1 = min(selectEndPoint[1], selectStartPoint[1])
        C2 = max(selectEndPoint[1], selectStartPoint[1])

        selectionMaskArray = np.empty_like(zoomInArray(XSecArray, globalZoom, globalX, globalY))
        selectionMaskArray[R1:R2, C1:C2] = 1
        selectionArray = np.copy(zoomInArray(XSecArray, globalZoom, globalX, globalY)[R1:R2, C1:C2])

        phase.set(5)


def pasteSelectionAtPoint(event, dataArray, canvas):
    global selectionArray, XSecArray

    if selectionArray is not None and len(selectionArray) and not inSelectMode:
        pasteRows = selectionArray.shape[0]
        pasteCols = selectionArray.shape[1]

        elementsInY = dataArray.shape[0]
        elementsInX = dataArray.shape[1]

        canvasHeight = canvas.winfo_height()
        canvasWidth  = canvas.winfo_width()

        dXY = min(canvasWidth / elementsInX, canvasHeight / elementsInY)
        startX = (canvasWidth  - dXY * elementsInX) / 2
        startY = (canvasHeight - dXY * elementsInY) / 2

        if (
            startX < event.x < canvasWidth  - startX
            and startY < event.y < canvasHeight - startY
        ):
            Col = int((event.x - startX) / dXY)
            Row = int((event.y - startY) / dXY)

            R1 = Row + globalY
            R2 = min(R1 + pasteRows, XSecArray.shape[0])
            C1 = Col + globalX
            C2 = min(C1 + pasteCols, XSecArray.shape[1])

            m = paste_mode.get()

            if m == 1:
                XSecArray[R1:R2, C1:C2] = np.where(
                    selectionArray[: R2 - R1, : C2 - C1] > 0,
                    selectionArray[: R2 - R1, : C2 - C1],
                    XSecArray[R1:R2, C1:C2],
                )
            elif m == 2:
                XSecArray[R1:R2, C1:C2] = selectionArray[: R2 - R1, : C2 - C1]
            elif 10 < m < 20:
                targetPhase = m - 10
                XSecArray[R1:R2, C1:C2] = np.where(
                    selectionArray[: R2 - R1, : C2 - C1] > 0,
                    targetPhase,
                    XSecArray[R1:R2, C1:C2],
                )

            printTheArray(dataArray, canvas)


def InterCode():
    global ic_currents, ic_materials, ic_custom_materials
    codeLines = text_input.get("1.0", END).split("\n")
    codeSteps, ic_currents, ic_materials, ic_custom_materials = ic.textToCode(codeLines)

    if codeSteps:
        for step in codeSteps:
            step[0](*step[1], XSecArray=XSecArray, dXmm=dXmm)

    redraw()


def getCanvas():
    global ic_currents, ic_materials, ic_custom_materials
    codeLines = text_input.get("1.0", END).split("\n")
    codeSteps, ic_currents, ic_materials, ic_custom_materials = ic.textToCode(codeLines)

    X = []
    Y = []
    circles = False

    if codeSteps:
        for step in codeSteps:
            tmp = step[0](*step[1], draw=False)
            if step[0] is ic.addCircle:
                circles = True
            X += [tmp[0], tmp[2]]
            Y += [tmp[1], tmp[3]]

        size = (max(X) - min(X), max(Y) - min(Y))
        sizes = [4, 2.5, 2, 1] if circles else [10, 5, 4, 2.5, 2, 1]
        xd = sizes[-1]
        for s in sizes:
            if size[0] % s == 0 and size[1] % s == 0:
                xd = s
                break

        elements = int(max(size[0] / xd, size[1] / xd))

        global dXmm, dYmm, XSecArray
        dXmm = dYmm = xd
        XSecArray = np.zeros([elements, elements])

        for step in codeSteps:
            step[0](*step[1], shift=(min(X), min(Y)), XSecArray=XSecArray, dXmm=dXmm)

    myEntryDx.delete(0, "end")
    myEntryDx.insert(0, str(dXmm))
    redraw()


######## End of functions definition ############
if __name__ == "__main__":
    master = ctk.CTk()
    master.title("Cross Section Designer 2")

    canvas_width  = 680
    canvas_height = 680

    master.resizable(width=True, height=True)

    temperatureArray = None

    mainSetup()

    master.bind("<Configure>", redraw)

    # ── Load icons ────────────────────────────────────────────────────────────
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
    up_icon_w         = PhotoImage(file="csdicons/up_white.png")
    down_icon_w       = PhotoImage(file="csdicons/down_white.png")
    left_icon_w       = PhotoImage(file="csdicons/left_white.png")
    right_icon_w      = PhotoImage(file="csdicons/right_white.png")
    zoom_in_icon      = PhotoImage(file="csdicons/zoomin.png")
    zoom_out_icon     = PhotoImage(file="csdicons/zoomout.png")
    up_icon           = PhotoImage(file="csdicons/up.png")
    down_icon         = PhotoImage(file="csdicons/down.png")
    left_icon         = PhotoImage(file="csdicons/left.png")
    right_icon        = PhotoImage(file="csdicons/right.png")

    # ── Menu bar ──────────────────────────────────────────────────────────────
    menu_bar = Menu(master,
        bg=BG_PANEL, fg=FG,
        activebackground=ACCENT, activeforeground="white",
        relief="flat", borderwidth=0,
    )

    def _menu(parent, **kw):
        return Menu(parent, tearoff=0,
            bg=BG_PANEL, fg=FG,
            activebackground=ACCENT, activeforeground="white", **kw)

    file_menu = _menu(menu_bar)
    file_menu.add_command(label="New geometry",        command=clearArrayAndDisplay)
    file_menu.add_separator()
    file_menu.add_command(label="Load from file",      command=loadArrayFromFile)
    file_menu.add_command(label="Import from picture", command=importArrayFromPicture)
    file_menu.add_separator()
    file_menu.add_command(label="Save to file",        command=saveArrayToFile)
    menu_bar.add_cascade(label="File", menu=file_menu)

    analyze_menu = _menu(menu_bar)
    analyze_menu.add_command(label="Power Losses ProSolver",            command=showMePro)
    analyze_menu.add_command(label="Electro Dynamic Forces",            command=showMeForces)
    analyze_menu.add_command(label="Equivalent Impedance Model",        command=showMeZ)
    analyze_menu.add_command(label="Equivalent Impedance Model 3f Shunt", command=showMeZ3f)
    menu_bar.add_cascade(label="Analyze...", menu=analyze_menu)

    geometry_menu = _menu(menu_bar)
    geometry_menu.add_command(label="Pattern",      command=showMeGeom)
    geometry_menu.add_command(label="Swap",         command=showReplacer)
    geometry_menu.add_separator()
    geometry_menu.add_command(label="Subdivide(+)", command=subdivideArray)
    geometry_menu.add_command(label="Simplify(-)",  command=simplifyArray)
    geometry_menu.add_separator()
    geometry_menu.add_command(label="Extend Canvas", command=extendArray)
    menu_bar.add_cascade(label="Geometry", menu=geometry_menu)

    view_menu = _menu(menu_bar)
    view_menu.add_command(label="Open CAD view window", command=displayArrayAsImage)
    menu_bar.add_cascade(label="View", menu=view_menu)

    master.config(menu=menu_bar)

    # ── IntVars (must exist before toolbar widgets) ───────────────────────────
    phase      = IntVar(value=1)
    paste_mode = IntVar(value=1)

    # ── Horizontal toolbar ────────────────────────────────────────────────────
    toolbar = ctk.CTkFrame(master, fg_color=TOOLBAR, corner_radius=0, height=76)
    toolbar.grid(row=0, column=0, columnspan=12, sticky="EW")
    toolbar.grid_propagate(False)

    def _section(parent, label):
        """Labelled card section for the toolbar."""
        outer = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=8,
                             border_width=1, border_color=SEP)
        outer.pack(side="left", padx=5, pady=6, fill="y")
        ctk.CTkLabel(outer, text=label, font=("Helvetica", 8, "bold"),
                     text_color=FG2).grid(row=0, column=0, columnspan=6,
                                          padx=6, pady=(3, 0), sticky="W")
        inner = ctk.CTkFrame(outer, fg_color=BG_PANEL, corner_radius=0)
        inner.grid(row=1, column=0, padx=4, pady=(0, 4))
        return inner

    def _icon_radio(parent, image, value, var, bg_color, row=0, col=0):
        rb = Radiobutton(
            parent, image=image, variable=var, value=value,
            indicatoron=0, height=32, width=32,
            bg=bg_color, activebackground=bg_color,
            selectcolor=ACCENT, highlightbackground=bg_color,
            relief="flat", cursor="hand2",
        )
        rb.grid(row=row, column=col, padx=2, pady=2)
        return rb

    def _icon_btn(parent, image, cmd, row=0, col=0, bw=28, bh=28, rep=False):
        kw = dict(repeatdelay=100, repeatinterval=100) if rep else {}
        btn = Button(
            parent, image=image, width=bw, height=bh,
            bg=BG_PANEL, activebackground=ACCENT,
            relief="flat", cursor="hand2",
            command=cmd, **kw,
        )
        btn.grid(row=row, column=col, padx=2, pady=2)
        return btn

    # Section 1 — Draw tool
    s_draw = _section(toolbar, "DRAW TOOL")
    _icon_radio(s_draw, A_icon_white,      1, phase, PH_A, row=0, col=0)
    _icon_radio(s_draw, B_icon_white,      2, phase, PH_B, row=0, col=1)
    _icon_radio(s_draw, C_icon_white,      3, phase, PH_C, row=0, col=2)
    _icon_radio(s_draw, cut_icon_white,    0, phase, PH_X, row=1, col=0)
    _icon_radio(s_draw, select_icon_white, 4, phase, PH_X, row=1, col=1)
    _icon_radio(s_draw, paste_icon_white,  5, phase, PH_X, row=1, col=2)

    # Section 2 — Paste mode
    s_paste = _section(toolbar, "PASTE MODE")
    _icon_radio(s_paste, paste_icon_white, 1,  paste_mode, PH_X, row=0, col=0)
    _icon_radio(s_paste, paste_icon_all,   2,  paste_mode, PH_X, row=0, col=1)
    _icon_radio(s_paste, paste_icon_A,    11,  paste_mode, PH_A, row=1, col=0)
    _icon_radio(s_paste, paste_icon_B,    12,  paste_mode, PH_B, row=1, col=1)
    _icon_radio(s_paste, paste_icon_C,    13,  paste_mode, PH_C, row=1, col=2)

    # Section 3 — Shift selected phase
    s_shift = _section(toolbar, "SHIFT PHASE")
    _icon_btn(s_shift, up_icon_w,    shiftU, row=0, col=1, rep=True)
    _icon_btn(s_shift, left_icon_w,  shiftL, row=1, col=0, rep=True)
    _icon_btn(s_shift, down_icon_w,  shiftD, row=1, col=1, rep=True)
    _icon_btn(s_shift, right_icon_w, shiftR, row=1, col=2, rep=True)

    # Section 4 — Grid size
    s_grid = _section(toolbar, "GRID SIZE")
    ctk.CTkLabel(s_grid, text="dx:", font=("Helvetica", 10), text_color=FG2).grid(
        row=0, column=0, padx=(6, 2), pady=4)
    myEntryDx = ctk.CTkEntry(s_grid, width=52, height=28,
                             fg_color=BG_WIDGET, border_color=SEP,
                             text_color=FG, font=("Helvetica", 11))
    myEntryDx.insert(0, str(dXmm))
    myEntryDx.grid(row=0, column=1, padx=2, pady=4)
    ctk.CTkLabel(s_grid, text="mm", font=("Helvetica", 10), text_color=FG2).grid(
        row=0, column=2, padx=(2, 6), pady=4)
    myEntryDx.bind("<Return>", setParameters)
    myEntryDx.bind("<FocusOut>", setParameters)

    analysisDX = ctk.CTkLabel(s_grid, text=f"dx: {dXmm} mm",
                              font=("Helvetica", 9), text_color=FG2)
    analysisDX.grid(row=1, column=0, columnspan=2, padx=6, pady=(0, 4), sticky="W")
    analysisDY = ctk.CTkLabel(s_grid, text=f"dy: {dYmm} mm",
                              font=("Helvetica", 9), text_color=FG2)
    analysisDY.grid(row=1, column=2, padx=6, pady=(0, 4), sticky="W")

    # Section 5 — View navigation
    s_nav = _section(toolbar, "VIEW NAV")
    _icon_btn(s_nav, zoom_in_icon,  zoomIn,  row=0, col=0, bw=32, bh=32)
    _icon_btn(s_nav, up_icon,       zoomU,   row=0, col=1, bw=28, bh=28, rep=True)
    _icon_btn(s_nav, zoom_out_icon, zoomOut, row=0, col=2, bw=32, bh=32)
    _icon_btn(s_nav, left_icon,     zoomL,   row=1, col=0, bw=28, bh=28, rep=True)
    _icon_btn(s_nav, down_icon,     zoomD,   row=1, col=1, bw=28, bh=28, rep=True)
    _icon_btn(s_nav, right_icon,    zoomR,   row=1, col=2, bw=28, bh=28, rep=True)

    # ── Main canvas ───────────────────────────────────────────────────────────
    w = Canvas(master, width=canvas_width, height=canvas_height)
    w.configure(background=CANVAS_BG, highlightthickness=0)
    w.grid(row=1, column=0, columnspan=8, rowspan=25,
           sticky=W + E + N + S, padx=2, pady=2)

    canvasElements = []
    shadowBox      = None

    inSelectMode     = False
    selectStartPoint = None
    selectEndPoint   = None
    selectShadowBox  = None
    selectionMaskArray = None
    selectionArray     = None

    # ── Inner-code panel ──────────────────────────────────────────────────────
    code_frame = ctk.CTkFrame(master, fg_color=BG_PANEL,
                              corner_radius=10, border_width=1, border_color=SEP)
    code_frame.grid(row=1, column=8, rowspan=24, columnspan=4,
                    sticky="NS", padx=(4, 8), pady=4)

    ctk.CTkLabel(code_frame, text="Inner-Code",
                 font=("Helvetica", 11, "bold"), text_color=FG).pack(pady=(8, 2))

    text_input = ctk.CTkTextbox(
        code_frame, height=340, width=200,
        fg_color=BG_WIDGET, text_color=FG,
        border_color=SEP, border_width=1,
        font=("Courier New", 11),
    )
    text_input.pack(padx=8, pady=(4, 6), fill="both", expand=True)

    ctk.CTkButton(code_frame, text="Execute InnerCode",
                  command=InterCode, fg_color=ACCENT, hover_color=ACCENT_H,
                  height=32, corner_radius=6).pack(
        padx=8, pady=(2, 4), fill="x")

    ctk.CTkButton(code_frame, text="Create new geometry",
                  command=getCanvas, fg_color=BG_WIDGET,
                  text_color=FG, hover_color=SEP,
                  border_width=1, border_color=SEP,
                  height=32, corner_radius=6).pack(
        padx=8, pady=(0, 8), fill="x")

    # ── Canvas bindings ───────────────────────────────────────────────────────
    w.bind("<Button 1>",      setPoint)
    w.bind("<Button 3>",      resetPoint)
    w.bind("<B1-Motion>",     setPoint)
    w.bind("<B3-Motion>",     resetPoint)
    w.bind("<Motion>",        shadowPoint)
    w.bind("<ButtonRelease-1>", endSelection)
    w.bind("<Button 2>",      showXsecArray)
    w.bind("<Left>",  zoomL)
    w.bind("<Right>", zoomR)
    w.bind("<Up>",    zoomU)
    w.bind("<Down>",  zoomD)

    # ── Status bar ────────────────────────────────────────────────────────────
    status = ctk.CTkFrame(master, fg_color=BG_WIDGET,
                          corner_radius=0, height=26, border_width=0)
    status.grid(row=26, column=0, columnspan=12, sticky="EW")
    status.grid_propagate(False)

    ctk.CTkLabel(
        status,
        text="  LMB: draw  •  RMB: erase  •  Arrow keys: pan  •  Analyze menu: run solver  •  Thermal analysis runs automatically",
        font=("Helvetica", 9), text_color=FG2,
    ).pack(side="left", padx=8)

    # ── Grid resize weights ───────────────────────────────────────────────────
    master.grid_rowconfigure(0, weight=0)
    master.grid_rowconfigure(12, weight=1)
    master.grid_columnconfigure(0, weight=1)
    master.grid_columnconfigure(8, weight=0)

    master.update()

    canvas_height = w.winfo_height()
    canvas_width  = w.winfo_width()

    printTheArray(dataArray=XSecArray, canvas=w)

    master.mainloop()
