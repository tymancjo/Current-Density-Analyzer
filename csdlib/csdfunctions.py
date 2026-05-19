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
from csdlib import csdmath as csdm
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
        myLog("The container is created")

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
        self.length = 1000
        self.dT = 0
        self.thermal_group = 0
        self.Fx = 0.0   # electromagnetic force x-component [N]
        self.Fy = 0.0   # electromagnetic force y-component [N]
        self.Fmag = 0.0 # |F| [N]

def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, *args, *kwargs)

def getCanvas(codeSteps):
    """This function is to determine the best parameters for the canvas
    based on the given geometry steps defined by the inner code."""

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
            # sizes = [4, 2.5, 2, 1]
            sizes = [2.5,2,1.5,1]
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

        # adding the defined cells to the geometry array
        for step in codeSteps:
            myLog(step)
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
                codeSteps, currents, materials, custom_materials, analysis_params = ic.textToCode(codeLines)

                XSecArray, dXmm, dYmm = getCanvas(codeSteps)
            except IOError:
                print("Error reading the file " + filename)
                sys.exit(1)

        else:
            myLog("reading from file :" + filename)
            XSecArray, dXmm, dYmm = loadObj(filename).restore()
            custom_materials  = {}
            analysis_params   = {}

        return XSecArray, dXmm, dYmm, currents, materials, custom_materials, analysis_params
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
    """BFS connected-component labelling with 4-connectivity.

    Each physically separate conductor body gets a unique integer ID in
    conductorsArr (1-based).  Conductors that share only a diagonal corner
    are treated as separate — consistent with the webCSD algorithm.

    Returns
    -------
    conductorsArr   : 2-D int array, same shape as XsecArr; cell value = conductor ID
    total_conductors: total number of distinct conductors found
    phaseCond       : list of lists — phaseCond[phase_idx] = [conductor IDs in that phase]
    """
    from collections import deque

    rows, cols = XsecArr.shape
    conductorsArr = np.zeros((rows, cols), dtype=int)
    num_phases = len(phases)
    phaseCond = [[] for _ in range(num_phases)]
    conductor_id = 0

    for r in range(rows):
        for c in range(cols):
            phase_val = int(XsecArr[r, c])
            if phase_val == 0 or conductorsArr[r, c] != 0:
                continue
            # Start a new BFS from this unvisited conductor cell
            conductor_id += 1
            phase_idx = phase_val - 1   # XsecArr is 1-indexed after normalisation
            if 0 <= phase_idx < num_phases:
                phaseCond[phase_idx].append(conductor_id)
            conductorsArr[r, c] = conductor_id
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if XsecArr[nr, nc] == phase_val and conductorsArr[nr, nc] == 0:
                            conductorsArr[nr, nc] = conductor_id
                            queue.append((nr, nc))

    return conductorsArr, conductor_id, phaseCond

def getPerymiter(vec, arr, dXmm, dYmm):
    if len(vec) > 0:
        row, col = int(vec[0, 0]), int(vec[0, 1])
        phase_id = arr[row, col]
        return csdm.getPerymiter(arr, dXmm, dYmm, phase_id=phase_id)
    return 0

def plot_the_geometry(DataArray, ax,cmap,  dXmm=10, dYmm=10, norm=None):

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

    return ax.imshow(DataArray, cmap=cmap, norm=norm)




def recreate_temperature_array(bars, xsec_array_shape):
    """
    This function recreates the XY cross section array with the temperature
    results. 

    """
    temperatures_array = np.zeros(xsec_array_shape)

    for bar in bars:
        for element in bar.elements:
            R = int(element[0])
            C = int(element[1])
            temperatures_array[R,C] = bar.dT

    return temperatures_array

def move_to_phase(bars, from_phase, to_phase):
    for bar in bars:
        if bar.phase == from_phase:
            bar.phase = to_phase


def compute_forces_for_bars(bars_data, currentsDraw, Isc_per_phase, length):
    """Compute Ampère electromagnetic forces on each conductor bar.

    Parameters
    ----------
    bars_data       : list of the_bar objects (elements must be populated)
    currentsDraw    : 2D numpy array of complex element currents (from recreateresultsArray)
    Isc_per_phase   : dict {phase_id: Isc_kA}  — signed short-circuit current per phase [kA]
    length          : busbar length [mm]

    Each bar's Fx, Fy, Fmag attributes are set in-place (forces in Newtons for the given length).
    """
    MU0_OVER_2PI = 2e-7   # μ₀/(2π) [H/m]
    length_m = length * 1e-3

    # ── Build flat element arrays ──────────────────────────────────────
    rows_list, cols_list, x_list, y_list, I_list, bar_list = [], [], [], [], [], []

    for bar_idx, bar in enumerate(bars_data):
        phase   = bar.phase
        Isc_kA  = float(Isc_per_phase.get(phase, 0.0))
        Isc_A   = Isc_kA * 1000.0

        elems = bar.elements
        if len(elems) == 0:
            continue

        # Magnitude of complex current per element within this bar
        I_mags = np.array([abs(currentsDraw[int(e[0]), int(e[1])]) for e in elems])
        sum_I  = I_mags.sum()

        for k, elem in enumerate(elems):
            if Isc_A == 0.0:
                I_force = 0.0
            elif sum_I > 1e-12:
                I_force = Isc_A * I_mags[k] / sum_I
            else:
                I_force = Isc_A / len(elems)

            rows_list.append(int(elem[0]))
            cols_list.append(int(elem[1]))
            x_list.append(elem[2] * 1e-3)   # mm → m
            y_list.append(elem[3] * 1e-3)
            I_list.append(I_force)
            bar_list.append(bar_idx)

    N = len(I_list)
    if N == 0:
        return bars_data

    ex = np.array(x_list,   dtype=np.float64)
    ey = np.array(y_list,   dtype=np.float64)
    eI = np.array(I_list,   dtype=np.float64)
    eb = np.array(bar_list, dtype=np.int32)

    # ── Vectorised O(N²) pairwise Ampère force ─────────────────────────
    dx = ex[np.newaxis, :] - ex[:, np.newaxis]   # (N,N) xj-xi  [m]
    dy = ey[np.newaxis, :] - ey[:, np.newaxis]   # (N,N) yj-yi  [m]
    d2 = dx**2 + dy**2                            # (N,N) distance² [m²]
    np.fill_diagonal(d2, np.inf)                  # zero self-force

    IiIj = eI[:, np.newaxis] * eI[np.newaxis, :] # (N,N) current products

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(d2 > 1e-20,
                          MU0_OVER_2PI * IiIj * length_m / d2,
                          0.0)                    # (N,N) force factor [N/m² * m = N]

    Fx_elem = (factor * dx).sum(axis=1)   # (N,) total Fx per element
    Fy_elem = (factor * dy).sum(axis=1)   # (N,) total Fy per element

    # ── Aggregate by bar ───────────────────────────────────────────────
    n_bars  = len(bars_data)
    bar_Fx  = np.zeros(n_bars, dtype=np.float64)
    bar_Fy  = np.zeros(n_bars, dtype=np.float64)
    np.add.at(bar_Fx, eb, Fx_elem)
    np.add.at(bar_Fy, eb, Fy_elem)

    for i, bar in enumerate(bars_data):
        bar.Fx   = float(bar_Fx[i])
        bar.Fy   = float(bar_Fy[i])
        bar.Fmag = float(np.hypot(bar.Fx, bar.Fy))

    return bars_data

