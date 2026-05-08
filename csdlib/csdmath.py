"""
this is the library of function to cover the math and calculation operations
in the CSD package.
"""

import numpy as np

try:
    from numba import njit
    use_njit = True

except ImportError:
    use_njit = False
    njit = None


# making the NUMBA decorators optional
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


# global constant to speed up loops
mi0 = 4 * np.pi * 1e-7
C1 = -np.log(2) / 3 + 13 / 12 - np.pi / 2
C2 = mi0 / (2 * np.pi)


# the functions
def solveTheEquation(admitanceMatrix, voltageVector):
    return np.matmul(admitanceMatrix, voltageVector)


def getGmatrix(input):
    return np.linalg.inv(input)


# @conditional_decorator(njit, use_njit)
def getImpedanceArray(
    distanceArray,
    freq,
    dXmm,
    dYmm,
    lenght=1000,
    temperature=20,
    sigma20C=58e6,
    temCoRe=3.9e-3,
    mi_r=1.0,
    mi_r_w=1.0,
    use_mi_array=False,
):
    """
    Calculate the array of impedance as complex values for each element
    Input:
    distanceArray -  array of distances beetween the elements in [mm]
    freq = frequency in Hz
    dXmm - size of element in x [mm]
    dYmm - size of element in Y [mm]
    lenght - analyzed lenght in [mm] /default= 1000mm
    temperature - temperature of the conductors in deg C / default = 20degC
    sigma20C - conductivity of conductor material in 20degC in [S] / default = 58MS (copper)
    temCoRe - temperature resistance coefficient / default is copper
    """
    
    omega = 2 * np.pi * freq
    
    # Handle mi_r and mi_r_w as either scalars or arrays
    # In vectorized mode we assume they are arrays of size N or scalars
    
    if use_mi_array:
        # Average mi_r_w for mutual inductance pairs
        mi_r_w_m = (mi_r_w.reshape(-1, 1) + mi_r_w.reshape(1, -1)) / 2
    else:
        mi_r_w_m = mi_r_w

    # Mutual Inductance for all pairs
    # Use a copy to avoid modifying the input and avoid log(0) on diagonal
    dist_safe = distanceArray.copy()
    for i in range(dist_safe.shape[0]):
        dist_safe[i, i] = 1.0
        
    M = getMutualInductance(dXmm, dYmm, lenght, dist_safe, mi_r_w_m)
    impedanceArray = 1j * omega * M
    
    # Self Inductance and Resistance for diagonal
    L_self = getSelfInductance(dXmm, dYmm, lenght, mi_r, mi_r_w)
    R = getResistance(dXmm, dYmm, lenght, temperature, sigma20C, temCoRe)
    
    # Combine diagonal: R + j*omega*L_self
    # R and L_self can be scalars or arrays
    diag_val = R + 1j * omega * L_self
    
    for i in range(impedanceArray.shape[0]):
        if isinstance(diag_val, np.ndarray):
            impedanceArray[i, i] = diag_val[i]
        else:
            impedanceArray[i, i] = diag_val
            
    return impedanceArray


@conditional_decorator(njit, use_njit)
def getSelfInductance(sizeX, sizeY, lenght, mi_r=1, mi_r_w=1):
    """
    Calculate the self inductance for the subconductor
    It assumes rectangular shape. If you want put for circular shape just
    make sizeX = sizeY = 2r

    Inputs:
    sizeX - width in [mm]
    sizeY - height in [mm]
    lenght - cinductor lenght in [mm]

    output
    L in [H]
    """
    srednica = (sizeX + sizeY) / 2
    a = srednica * 1e-3
    l = lenght * 1e-3

    # New formula to use both mi_r for material and the surroundings
    r = a / 2.0
    mu_o = mi0 * mi_r_w  # przenikalność magnetyczna ośrodka

    L_m = (mu_o / (2 * np.pi)) * (np.log(2 * l / r) - 1 + mi_r / 4)
    L = L_m * l

    return L


@conditional_decorator(njit, use_njit)
def getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    """
    Calculate the resistance of the al'a square shape in given temperature
    All dimensions in mm
    temperature in deg C

    output:
    Resistance in Ohm
    """
    return (lenght / (sizeX * sizeY * sigma20C)) * 1e3 * (1 + temCoRe * (temp - 20))


@conditional_decorator(njit, use_njit)
def getMutualInductance(sizeX, sizeY, lenght, distance, mi_r_w=1):
    """
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
    """
    srednica = (sizeX + sizeY) / 2

    a = 0.5 * srednica * 1e-3
    l = lenght * 1e-3
    d = distance * 1e-3

    # formula by:
    # https://pdfs.semanticscholar.org/b0f4/eff92e31d4c5ff42af4a873ebdd826e610f5.pdf
    M = (mi0 * mi_r_w * l / (2 * np.pi)) * (
        np.log((l + np.sqrt(l**2 + d**2)) / d) - np.sqrt(1 + (d / l) ** 2) + d / l
    )

    return M


@conditional_decorator(njit, use_njit)
def getResistanceArray(
    elementsVector,
    dXmm,
    dYmm,
    lenght=1000,
    temperature=20,
    sigma20C=58e6,
    temCoRe=3.9e-3,
    sigma_array=None,
    alpha_array=None,
):
    """
    Calculate the array of resistance values for each element
    Input:
    elementsVector - The elements vector as delivered by arrayVectorize
    dXmm - size of element in x [mm]
    dYmm - size of element in Y [mm]
    lenght - analyzed lenght in [mm] /default= 1000mm
    temperature - temperature of the conductors in deg C / defoult = 20degC
    sigma20C - conductivity of conductor material in 20degC in [S] / default = 58MS (copper)
    temCoRe - temperature resistance coeficcient / default is copper
    """

    if sigma_array is not None and alpha_array is not None:
        # We need to map material properties to each element
        # elementsVector[:, 4] contains the phase/material number
        # We need to ensure we use the correct mapping
        
        # To handle this vectorized, we pre-allocate properties
        s = np.zeros(elementsVector.shape[0])
        a = np.zeros(elementsVector.shape[0])
        
        for i in range(elementsVector.shape[0]):
            mat_idx = int(elementsVector[i, 4])
            s[i] = sigma_array[mat_idx]
            a[i] = alpha_array[mat_idx]
        
        return (lenght / (dXmm * dYmm * s)) * 1e3 * (1 + a * (temperature - 20))
    else:
        res = (lenght / (dXmm * dYmm * sigma20C)) * 1e3 * (1 + temCoRe * (temperature - 20))
        return np.full(elementsVector.shape[0], res)


@conditional_decorator(njit, use_njit)
def getPerymiter(arr, dXmm, dYmm, phase_id=None):
    """
    This function returns the area perymiter lenght for given
    conducting elements in the array.
    It counts edges that are adjacent to empty space (0).
    
    Inputs:
    arr - array that describe the geometry shape
    dXmm - element size in x diretion
    dYmm - element size in y diretion
    phase_id - if provided, only perymiter of this phase is calculated.
               If None, perymiter of all conductors is calculated.
    """
    
    if phase_id is not None:
        b = (arr == phase_id)
    else:
        b = (arr > 0)
    
    rows, cols = b.shape
    perymiter = 0.0
    
    for r in range(rows):
        for c in range(cols):
            if b[r, c]:
                # Check 4 neighbors
                # Top
                if r == 0 or arr[r-1, c] == 0:
                    perymiter += dXmm
                # Bottom
                if r == rows - 1 or arr[r+1, c] == 0:
                    perymiter += dXmm
                # Left
                if c == 0 or arr[r, c-1] == 0:
                    perymiter += dYmm
                # Right
                if c == cols - 1 or arr[r, c+1] == 0:
                    perymiter += dYmm
                    
    return perymiter

@conditional_decorator(njit, use_njit)
def getDistancesArray(inputVector):
    """
    This function calculate the array of distances between every conductors
    element
    Input:
    the vector of conductor elements as delivered by n_vectorizeTheArray
    """
    # Using numpy broadcasting for vectorization
    coords = inputVector[:, 2:4]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distanceArray = np.sqrt(np.sum(diff**2, axis=-1))
    
    return distanceArray


@conditional_decorator(njit, use_njit)
def get_mi_weighted(XsecArr, mi_r_siatka, dXmm, delta=10):
    """
    Oblicza zastępcze mi_r dla całej macierzy, uwzględniając sąsiedztwo ±delta komórek.
    Wersja zoptymalizowana pod Numba.

    Args:
        mi_r_siatka: Macierz 2D z wartościami mi_r dla każdego kwadratu.
        dXmm: Długość boku kwadratu w mm.

    Returns:
        Macierz z zastępczymi wartościami mi_r.
    """
    delta = int(delta / dXmm)
    rows, cols = mi_r_siatka.shape
    zastepcze_mi_r_macierz = np.ones((rows, cols))
    
    nonzero_rows, nonzero_cols = np.nonzero(XsecArr)

    # Iterujemy tylko po niezerowych elementach XsecArr
    for k in range(len(nonzero_rows)):
        x, y = nonzero_rows[k], nonzero_cols[k]
        
        sum_wagi = 0.0
        sum_mi_r_wagi = 0.0
        
        # Okno wokół (x, y)
        for i in range(max(0, x - delta), min(rows, x + delta + 1)):
            for j in range(max(0, y - delta), min(cols, y + delta + 1)):
                if i == x and j == y:
                    continue  # Pomijamy bieżący kwadrat

                odleglosc = np.sqrt((i - x)**2 + (j - y)**2) * dXmm
                waga = 1 / (odleglosc + 0.0001)
                
                sum_wagi += waga
                sum_mi_r_wagi += waga * mi_r_siatka[i, j]
        
        if sum_wagi > 0:
            zastepcze_mi_r_macierz[x, y] = sum_mi_r_wagi / sum_wagi
            
    return zastepcze_mi_r_macierz


@conditional_decorator(njit, use_njit)
def get_mi_averaged(XsecArr, mi_r_array, dXmm, delta=10):
    """
    This is different approach to the the average mi_r calculation.
    The idea is:
        we start from a R,C coordinate,
        we do delta times the average value of sub array around the R,C
        each time taking bigger area around.
    """

    rows, cols = mi_r_array.shape
    mi_r_average = np.ones((rows, cols))

    nonzero_indices = np.nonzero(XsecArr)
    elements_coordinates = np.stack(nonzero_indices, axis=-1)

    sizes = [1, 3, 7] + [
        s for s in range(10, min(rows, cols), min(rows, cols) // delta)
    ]

    for R, C in elements_coordinates:
        mi_r_av = 0
        for s in sizes:
            r_top = max(R - s, 0)
            r_btm = min(R + s, rows)

            c_left = max(C - s, 0)
            c_right = min(C + s, cols)

            sub_array_num_elements = (r_btm-r_top)*(c_right-c_left) - 1

            if sub_array_num_elements:
                mi_r_av += (np.sum(mi_r_array[r_top:r_btm, c_left:c_right]) - mi_r_array[R,C]) / sub_array_num_elements
        mi_r_average[R, C] = mi_r_av / len(sizes)

    return mi_r_average


def arrayVectorize(inputArray, phaseNumber, dXmm, dYmm):
    """
    Desription:
    This function returns vector of 4 dimension vectors that deliver

    input:
    inputArray = 3D array thet describe by 1's position of
    conductors in cross section
    dXmm - size of each element in X direction [mm]
    dYmm - size of each element in Y direction [mm]
    Output:
    [0,1,2,3] - 4 elements vector for each element, where:

    0 - Original inputArray geometry origin Row for the set cell
    1 - Original inputArray geometry origin Col for the set cell
    2 - X position in mm of the current element
    3 - Y position in mm of the current element

    Number of such [0,1,2,3] elements is equal to the number of defined
    conductor cells in geometry

    """
    # Let's check the size of the array
    elementsInY = inputArray.shape[0]
    elementsInX = inputArray.shape[1]

    # lets define the empty vectorArray
    vectorArray = []

    # lets go for each input array position and check if is set
    # and if yes then put it into putput vectorArray
    for Row in range(elementsInY):
        for Col in range(elementsInX):
            if inputArray[Row][Col] == phaseNumber:
                # Let's calculate the X and Y coordinates
                coordinateY = (0.5 + Row) * dYmm
                coordinateX = (0.5 + Col) * dXmm

                vectorArray.append([Row, Col, coordinateX, coordinateY, phaseNumber])

    return np.array(vectorArray)


def arraySlicer(inputArray, subDivisions=2):
    """
    This function increase the resolution of the cross section array
    inputArray -  oryginal geometry matrix
    subDivisions -  number of subdivisions / factor of increase of resoluttion / default = 2
    """
    return inputArray.repeat(subDivisions, axis=0).repeat(subDivisions, axis=1)


def getComplexModule(x):
    """
    returns the module of complex number
    input: x - complex number
    if not a complex number is given as parameter then it return the x diretly

    """
    if isinstance(x, complex):
        return np.absolute(x)
    else:
        return x


# Function that put back together the solution vectr back to represent the cross
# section shape array
def recreateresultsArray(
    elementsVector, resultsVector, initialGeometryArray, dtype=float
):
    """
    Functions returns recreate cross section array with mapperd solution results
    Inputs:
    elementsVector - vector of crossection elements as created by the n_arrayVectorize
    resultsVector - vectr with results values calculated base on the elementsVector
    initialGeometryArray - the array that contains the cross section geometry model
    """
    localResultsArray = np.zeros((initialGeometryArray.shape), dtype=dtype)

    for vectorIndex, result in enumerate(resultsVector):
        localResultsArray[
            int(elementsVector[vectorIndex][0]), int(elementsVector[vectorIndex][1])
        ] = result

    return localResultsArray
