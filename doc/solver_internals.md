# CSD Solver Internals — Math, Pre-processing, and Post-processing

This document is a complete, code-annotated walkthrough of the numerical engine.
It covers every calculation stage from raw geometry input to power-loss output,
referencing the actual Python functions and showing the exact formulae in use.

**Source files covered**

| File | Role |
|---|---|
| `csdlib/csdmath.py` | All numerical primitives (inductance, resistance, distances, vectorisation) |
| `csdlib/csdsolve.py` | Solver functions (impedance → admittance → solve → normalise → losses) |
| `csdlib/csdfunctions.py` | Pre/post utilities (load, canvas fit, conductor detection, thermal post-process) |
| `csdlib/csdos.py` | `Material` dataclass and materials-file reader |

---

## Table of Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Data structures](#2-data-structures)
3. [Pre-processing](#3-pre-processing)
   - 3.1 Mesh refinement — `arraySlicer`
   - 3.2 Element vectorisation — `arrayVectorize`
   - 3.3 Pairwise distances — `getDistancesArray`
   - 3.4 Material arrays (magnetic solver)
   - 3.5 Weighted magnetic permeability — `get_mi_weighted`
4. [Impedance matrix assembly](#4-impedance-matrix-assembly)
   - 4.1 Element resistance — `getResistance`
   - 4.2 Self-inductance — `getSelfInductance`
   - 4.3 Mutual inductance — `getMutualInductance`
   - 4.4 Full assembly — `getImpedanceArray`
5. [Admittance matrix](#5-admittance-matrix)
6. [Solve algorithm](#6-solve-algorithm)
   - 6.1 Complex current targets
   - 6.2 Trial voltage vector
   - 6.3 First solve
   - 6.4 Voltage correction
   - 6.5 Second (final) solve
   - 6.6 Magnitude normalisation
7. [Power losses post-processing](#7-power-losses-post-processing)
8. [Result array reconstruction](#8-result-array-reconstruction)
9. [Thermal solver (bar-level)](#9-thermal-solver-bar-level)
10. [Conductor detection](#10-conductor-detection)
11. [Solver variants](#11-solver-variants)
12. [Complexity and performance notes](#12-complexity-and-performance-notes)

---

## 1. Pipeline overview

```
.ic file / .csd file
        │
        ▼
  loadTheData()          parse geometry, build XSecArray [rows × cols]
        │
        ▼ (optional)
  arraySlicer()          refine mesh until cell size ≤ requested maximum
        │
        ▼
  solve_with_magnetic()  top-level solver (used by CLI and Pro solver)
    │
    ├── build mi_r_array       map phase IDs → μr values
    ├── get_mi_weighted()      compute per-cell effective μr_w
    ├── arrayVectorize()       XSecArray → elementsVector  [N × 5]
    ├── getDistancesArray()    elementsVector → distanceArray  [N × N]
    ├── getImpedanceArray()    distanceArray → Z  [N × N complex]
    ├── getGmatrix()           Z → G = Z⁻¹  [N × N complex]
    ├── first solve            currentVector = G @ voltageVector
    ├── voltage correction     scale U per phase to hit target I
    ├── second solve           currentVector = G @ correctedVoltageVector
    ├── magnitude normalise    exact Irms enforcement
    └── getResistanceArray()   → powerLossesVector = R * |I|²
        │
        ▼
  recreateresultsArray()   1-D results → 2-D display arrays
```

---

## 2. Data structures

### XSecArray — the geometry grid

A 2-D NumPy array of shape `(rows, cols)` where each cell holds an integer:

- `0` → empty (air/void)
- `1, 2, 3, …` → phase/conductor ID

Cell size is `dXmm × dYmm` mm (typically square, so `dXmm == dYmm`).
The physical x-coordinate of cell `(r, c)` is `(c + 0.5) * dXmm`; y is `(r + 0.5) * dYmm`.

### elementsVector — the flat conductor list

Shape `(N, 5)`. Each row represents one conducting cell:

| Column | Content |
|---|---|
| `[:,0]` | grid row index (integer) |
| `[:,1]` | grid column index (integer) |
| `[:,2]` | physical x-coordinate of cell centre [mm] |
| `[:,3]` | physical y-coordinate of cell centre [mm] |
| `[:,4]` | phase ID |

N is the total number of non-zero cells across all phases. The rows are ordered phase-by-phase (all cells of phase 1 first, then phase 2, etc.).

### distanceArray — pairwise distances

Shape `(N, N)`. Element `[i, j]` is the Euclidean distance in mm between the centres of cells i and j. The diagonal is 0 (but temporarily overwritten to 1.0 before the log computation to avoid singularity).

### Z and G — the system matrices

Shape `(N, N)` complex. Z is the impedance matrix; G = Z⁻¹ is the admittance matrix. Both are full (dense) matrices — no sparsity is exploited.

### materials_Xsec_array — per-cell material tensor

Shape `(rows, cols, 4)`. The four layers are:

| Index | Content |
|---|---|
| `0` (`idx_sigma`) | electrical conductivity σ [S/m] |
| `1` (`idx_alpha`) | temperature coefficient of resistance α [1/K] |
| `2` (`idx_m_r`) | material relative permeability μr |
| `3` (`idx_m_r_w`) | weighted effective permeability of the surrounding medium μr_w |

This array is built in `solve_with_magnetic` before element vectorisation so that σ, α, μr, and μr_w can be looked up per element from `elementsVector[:,0:2]` as integer row/col indices.

---

## 3. Pre-processing

### 3.1 Mesh refinement — `arraySlicer` (`csdmath.py:356`)

```python
def arraySlicer(inputArray, subDivisions=2):
    return inputArray.repeat(subDivisions, axis=0).repeat(subDivisions, axis=1)
```

Each call doubles (or N-tuples) the resolution in both dimensions by repeating every row and column. After one call with `subDivisions=2`, a 10 mm cell becomes four 5 mm cells all carrying the same phase ID.

The CLI loop:
```python
while dXmm > config["size"]:
    XSecArray = csdm.arraySlicer(inputArray=XSecArray, subDivisions=2)
    dXmm /= 2
    dYmm /= 2
```

Halving cell size quadruples N, which multiplies Z-inversion time by ~8×.

### 3.2 Element vectorisation — `arrayVectorize` (`csdmath.py:346`)

```python
def arrayVectorize(inputArray, phaseNumber, dXmm, dYmm):
    positions = np.argwhere(inputArray == phaseNumber)   # shape (k, 2): [row, col]
    if len(positions) == 0:
        return np.empty((0, 5))
    coordinateX = (0.5 + positions[:, 1]) * dXmm        # cell centre x [mm]
    coordinateY = (0.5 + positions[:, 0]) * dYmm        # cell centre y [mm]
    phase_col = np.full(len(positions), float(phaseNumber))
    return np.column_stack([positions.astype(float), coordinateX, coordinateY, phase_col])
```

Called once per phase, then concatenated into `elementsVector = np.concatenate(vPh)`.

The `0.5 + index` offset places the coordinate at the cell centre rather than the edge.

### 3.3 Pairwise distances — `getDistancesArray` (`csdmath.py:249`)

```python
def getDistancesArray(inputVector):
    coords = inputVector[:, 2:4]                          # shape (N, 2): x,y in mm
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    distanceArray = np.sqrt(np.sum(diff**2, axis=-1))    # (N, N) mm
    return distanceArray
```

Pure NumPy broadcasting — no explicit loops. The result is a symmetric matrix with zeros on the diagonal.

### 3.4 Material arrays (`solve_with_magnetic`, `csdsolve.py:518–570`)

```python
# Step 1: build μr grid (integer array, same shape as XSecArray)
mi_r_array = np.copy(XsecArr).astype(int)
mi_r_array[mi_r_array == 0] = 1        # air → μr = 1

for phase in list_of_phases:
    mi_r = phases_materials[phase_index].mi_r
    mi_r_array[mi_r_array == phase] = mi_r   # replace ID with actual μr

# Step 2: compute weighted μr for each cell (see 3.5)
mi_r_weighted_array = csdm.get_mi_weighted(XsecArr, mi_r_array, dXmm, delta=250)

# Step 3: build per-cell lookup tensor
for phase in list_of_phases:
    mask = XsecArr == phase
    p_idx = phase_index_dict[phase]
    mat = phases_materials[p_idx]
    materials_Xsec_array[mask, idx_sigma]   = mat.sigma
    materials_Xsec_array[mask, idx_alpha]   = mat.alpha
    materials_Xsec_array[mask, idx_m_r]     = mi_r_array[mask]
    materials_Xsec_array[mask, idx_m_r_w]   = mi_r_weighted_array[mask]

# Step 4: extract 1-D arrays indexed by element order
rows_ev = elementsVector[:, 0].astype(int)
cols_ev = elementsVector[:, 1].astype(int)
sigma_array   = materials_Xsec_array[rows_ev, cols_ev, idx_sigma]    # shape (N,)
alpha_array   = materials_Xsec_array[rows_ev, cols_ev, idx_alpha]    # shape (N,)
mi_r_array    = materials_Xsec_array[rows_ev, cols_ev, idx_m_r]      # shape (N,)
mi_r_w_array  = materials_Xsec_array[rows_ev, cols_ev, idx_m_r_w]   # shape (N,)
```

The result is four 1-D arrays of length N, aligned element-for-element with `elementsVector`. These feed directly into `getImpedanceArray` without any further lookup.

### 3.5 Weighted magnetic permeability — `get_mi_weighted` (`csdmath.py:265`)

This function answers the question: *"What effective permeability does the magnetic flux from element (r, c) see in its surroundings?"*

```python
def get_mi_weighted(XsecArr, mi_r_siatka, dXmm, delta=10):
    delta_cells = int(delta / dXmm)            # delta mm → number of cells
    nonzero_rows, nonzero_cols = np.nonzero(XsecArr)

    for k in range(len(nonzero_rows)):
        x, y = nonzero_rows[k], nonzero_cols[k]
        sum_wagi = 0.0
        sum_mi_r_wagi = 0.0

        # iterate over ± delta_cells window around (x, y)
        for i in range(max(0, x - delta_cells), min(rows, x + delta_cells + 1)):
            for j in range(max(0, y - delta_cells), min(cols, y + delta_cells + 1)):
                if i == x and j == y:
                    continue                    # skip self
                distance_mm = sqrt((i-x)**2 + (j-y)**2) * dXmm
                weight = 1.0 / (distance_mm + 1e-4)    # inverse distance
                sum_wagi     += weight
                sum_mi_r_wagi += weight * mi_r_siatka[i, j]

        zastepcze_mi_r_macierz[x, y] = sum_mi_r_wagi / sum_wagi
```

The result `mi_r_weighted_array[r, c]` is the inverse-distance-weighted average of the μr values of all cells within `delta` mm. The ε = 1e-4 mm prevents division by zero for immediately adjacent cells.

The call site uses `delta=250` mm, meaning the influence radius is 250 mm — effectively the whole cross-section for typical busbar arrangements.

This value is used as `μr_w` (the permeability of the *medium around* a conductor) in inductance calculations. The material's own permeability (its internal inductance contribution) is kept separately as `mi_r`.

---

## 4. Impedance matrix assembly

### 4.1 Element resistance — `getResistance` (`csdmath.py:132`)

For a single scalar call (used during diagonal assembly):

```python
def getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    return (lenght / (sizeX * sizeY * sigma20C)) * 1e3 * (1 + temCoRe * (temp - 20))
```

**Derivation:**

The cell cross-section area is $A = \text{sizeX} \times \text{sizeY}\ \text{mm}^2 = \text{sizeX} \times \text{sizeY} \times 10^{-6}\ \text{m}^2$.

The conductor length is $l = \text{lenght}\ \text{mm} = \text{lenght} \times 10^{-3}\ \text{m}$.

The DC resistance at 20 °C:
$$R_{20} = \frac{l}{A \cdot \sigma_{20}} = \frac{\text{lenght} \times 10^{-3}}{\text{sizeX} \times \text{sizeY} \times 10^{-6} \times \sigma_{20}}$$

Simplifying $10^{-3} / 10^{-6} = 10^3$:
$$R_{20} = \frac{\text{lenght}}{\text{sizeX} \times \text{sizeY} \times \sigma_{20}} \times 10^3 \quad [\Omega]$$

Temperature correction (linear):
$$R(T) = R_{20} \cdot \left(1 + \alpha\,(T - 20)\right)$$

For copper at 20 °C with 5 mm cells and 1000 mm length:
```
R = (1000 / (5 * 5 * 56e6)) * 1e3 * 1.0 = 7.14 × 10⁻⁴ Ω
```

When called via `getResistanceArray` with per-element material arrays:
```python
def getResistanceArray(..., sigma_array=None, alpha_array=None):
    if sigma_array is not None and alpha_array is not None:
        return (lenght / (dXmm * dYmm * sigma_array)) * 1e3 * (1 + alpha_array * (temperature - 20))
    else:
        return np.full(N, scalar_R)
```

Here `sigma_array` and `alpha_array` are shape-(N,) arrays, so the formula is broadcast element-wise, giving a shape-(N,) resistance vector.

### 4.2 Self-inductance — `getSelfInductance` (`csdmath.py:102`)

```python
def getSelfInductance(sizeX, sizeY, lenght, mi_r=1, mi_r_w=1):
    srednica = (sizeX + sizeY) / 2          # average dimension → "equivalent diameter" [mm]
    a  = srednica * 1e-3                    # [m]
    r  = a / 2.0                            # equivalent radius [m]
    l  = lenght * 1e-3                      # [m]
    mu_o = mi0 * mi_r_w                     # permeability of surrounding medium [H/m]
    L_m = (mu_o / (2*pi)) * (log(2*l/r) - 1 + mi_r/4)
    L   = L_m * l                           # [H]
    return L
```

**Formula source:** Neumann formula for a straight finite-length cylindrical conductor.

The self-inductance of a round wire of radius r and length l is:
$$L = \frac{\mu_0 \mu_{r,w}}{2\pi} \left[\ln\!\frac{2l}{r} - 1 + \frac{\mu_r}{4}\right] \cdot l \quad \text{[H]}$$

- The $\ln(2l/r) - 1$ term is the *external* inductance (flux outside the wire).
- The $\mu_r / 4$ term is the *internal* inductance (flux inside the conductor material), scaled by the material's relative permeability.
- $\mu_{r,w}$ scales the external inductance by the permeability of the surrounding medium.

For a square cell with `sizeX ≠ sizeY` the code uses the average dimension as the equivalent round wire diameter. This is an approximation.

Example — 5 mm copper cell, 1000 mm long, μr = μr_w = 1:
```
r = 2.5e-3 m
l = 1.0 m
L = (4π×10⁻⁷ / 2π) * (ln(800) - 1 + 0.25) * 1.0
  = 2×10⁻⁷ * (6.685 - 1 + 0.25)
  = 2×10⁻⁷ * 5.935
  ≈ 1.19 μH
```

### 4.3 Mutual inductance — `getMutualInductance` (`csdmath.py:144`)

```python
def getMutualInductance(sizeX, sizeY, lenght, distance, mi_r_w=1):
    srednica = (sizeX + sizeY) / 2
    a = 0.5 * srednica * 1e-3               # radius [m] (unused in the formula below)
    l = lenght * 1e-3                       # [m]
    d = distance * 1e-3                     # centre-to-centre distance [m]
    M = (mi0 * mi_r_w * l / (2*pi)) * (
            log((l + sqrt(l**2 + d**2)) / d)
            - sqrt(1 + (d/l)**2)
            + d/l
        )
    return M                                # [H]
```

**Formula source:** Exact Neumann formula for mutual inductance between two parallel, coaxial, finite-length filaments separated by distance d:

$$M = \frac{\mu_0 \mu_{r,w} \cdot l}{2\pi} \left[ \ln\frac{l + \sqrt{l^2 + d^2}}{d} - \sqrt{1 + \left(\frac{d}{l}\right)^2} + \frac{d}{l} \right]$$

This is exact for point filaments; treating each cell as a single filament at its centre is the main approximation of the model. The error decreases as cell size decreases relative to inter-cell spacing.

Note: `a` (radius) is computed but not used in the returned value — it was present in an earlier formula variant. The current formula depends only on `l` and `d`.

When `use_mi_array=True` (the magnetic solver), `mi_r_w` is a shape-(N, N) matrix:
```python
mi_r_w_m = (mi_r_w.reshape(-1,1) + mi_r_w.reshape(1,-1)) / 2
```
The effective medium permeability for the i–j pair is the arithmetic mean of the per-element weighted permeabilities.

### 4.4 Full assembly — `getImpedanceArray` (`csdmath.py:44`)

```python
def getImpedanceArray(distanceArray, freq, dXmm, dYmm, lenght, temperature,
                      sigma20C, temCoRe, mi_r=1.0, mi_r_w=1.0, use_mi_array=False):

    omega = 2 * pi * freq

    # --- off-diagonal: mutual inductances → Z_ij = jω·M_ij ---
    if use_mi_array:
        mi_r_w_m = (mi_r_w.reshape(-1,1) + mi_r_w.reshape(1,-1)) / 2   # (N,N)
    else:
        mi_r_w_m = mi_r_w

    dist_safe = distanceArray.copy()
    np.fill_diagonal(dist_safe, 1.0)     # avoid log(0) on diagonal
    M = getMutualInductance(dXmm, dYmm, lenght, dist_safe, mi_r_w_m)   # (N,N)
    impedanceArray = 1j * omega * M      # (N,N) complex

    # --- diagonal: self-impedance Z_ii = R_i + jω·L_i ---
    L_self = getSelfInductance(dXmm, dYmm, lenght, mi_r, mi_r_w)       # scalar or (N,)
    R      = getResistance(dXmm, dYmm, lenght, temperature, sigma20C, temCoRe)
    diag_val = R + 1j * omega * L_self
    np.fill_diagonal(impedanceArray, diag_val)

    return impedanceArray   # shape (N,N) complex
```

**The resulting Z matrix structure:**

$$Z = \begin{pmatrix}
R_1 + j\omega L_1 & j\omega M_{12} & j\omega M_{13} & \cdots \\
j\omega M_{21} & R_2 + j\omega L_2 & j\omega M_{23} & \cdots \\
\vdots & & \ddots & \\
\end{pmatrix}$$

When `use_mi_array=True`, the diagonal values `diag_val` is a shape-(N,) array (different σ, α, μr per element), filled with `np.fill_diagonal` which accepts a 1-D array.

When `use_mi_array=False`, `diag_val` is a scalar and all diagonal entries are identical (single-material mode).

---

## 5. Admittance matrix

```python
def getGmatrix(input):
    return np.linalg.inv(input)    # shape (N,N) complex
```

**G = Z⁻¹**

This is the full dense matrix inversion via LAPACK (LU decomposition under the hood). It is O(N³) in time and O(N²) in memory. For N = 500 the matrix is 500² × 16 bytes ≈ 4 MB; for N = 2000 it is 64 MB.

G is computed **once** and reused for both the trial and corrected solve passes. This is the most computationally expensive single step.

---

## 6. Solve algorithm

The following describes `solve_with_magnetic` (`csdsolve.py:444`), which is the production solver. The structure is identical in `solve_multi_system` and `solve_system`.

### 6.1 Complex current targets

```python
# currents input: list of [Irms, phi_degrees] per phase
I = []
for i in currents:
    Imod = float(i[0])
    Phi  = float(i[1]) * pi / 180
    I.append(Imod * (cos(Phi) + 1j * sin(Phi)))
```

`I[n]` is the complex phasor of the target total current for phase n:
$$I_n = I_{rms,n} \cdot e^{j\phi_n} = I_{rms,n}(\cos\phi_n + j\sin\phi_n)$$

For balanced three-phase at 2300 A:
```
I[0] = 2300∠0°   = 2300 + 0j
I[1] = 2300∠120° = -1150 + 1993j
I[2] = 2300∠240° = -1150 - 1993j
```

### 6.2 Trial voltage vector

```python
U = []
for i in currents:
    Phi = float(i[1]) * pi / 180
    U.append(cos(Phi) + sin(Phi) * 1j)    # unit phasor at correct angle
```

Each phase starts with a unit-magnitude voltage at the correct phase angle. The actual magnitude will be corrected in step 6.4.

```python
# Build full voltage vector: N complex values
voltageVector = concatenate([ones(n_elem, dtype=complex) * u
                             for n_elem, u in zip(elementsPhase, U)])
```

All elements of phase n receive the same voltage phasor `U[n]`. This is the critical assumption: *a phase is driven by a single, uniform voltage source applied equally to all its cells*. The impedance coupling between cells then determines how the current distributes.

### 6.3 First solve

```python
currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)
# = np.matmul(G, voltageVector)    shape (N,) complex
```

The matrix-vector product **G · U** gives the complex current through each element. This is O(N²), cheap compared to the inversion.

```python
# Split result back into per-phase slices
currentsPh = []
start = 0
for n_elem in elementsPhase:
    currentsPh.append(currentVector[start : start + n_elem])
    start += n_elem

# Sum each phase
I_results = [np.sum(cPh) for cPh in currentsPh]
```

`I_results[n]` is the complex phasor of the total current computed for phase n under the trial voltage.

### 6.4 Voltage correction

```python
for n, (i_r, i, u) in enumerate(zip(I_results, I, U)):
    if i_r:
        this_Z = u / i_r       # effective impedance: Z_eff = U_trial / I_computed
        U[n]   = this_Z * i    # scale: U_new = Z_eff * I_target
```

**Why this works:**

The system is linear, so if voltage `U_trial` produced current `I_computed`, then to produce `I_target` we need:

$$U_{\text{new}} = U_{\text{trial}} \cdot \frac{I_{\text{target}}}{I_{\text{computed}}}$$

Written as the code does it:
$$Z_{\text{eff}} = \frac{U_{\text{trial}}}{I_{\text{computed}}}, \quad U_{\text{new}} = Z_{\text{eff}} \cdot I_{\text{target}}$$

Both forms are identical. The complex ratio preserves the phase information — the new voltage will have the correct magnitude and phase to drive the target current.

This correction requires only one application of the admittance matrix (already computed) — no re-inversion.

### 6.5 Second (final) solve

```python
voltageVector = concatenate([ones(n_elem, dtype=complex) * u
                             for n_elem, u in zip(elementsPhase, U)])
currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)
```

G is reused. The corrected voltage produces a current distribution that closely matches the target currents.

### 6.6 Magnitude normalisation

```python
modI = [np.abs(np.sum(cPh)) for cPh in currentsPh]   # total magnitude per phase

for n, (mod_i, cPh, i) in enumerate(zip(modI, currentsPh, currents)):
    if mod_i != 0 and i[0] != 0:
        currentsPh[n] = cPh * i[0] / mod_i
```

After the second solve the total per-phase current magnitude may still differ slightly from the target (due to the iterative approximation). This step scales the entire current vector of each phase by the ratio `Irms_target / |Irms_computed|`.

**What this preserves:** the *relative distribution* within the phase — the ratio between currents in different elements is unchanged. Only the global magnitude is corrected.

**What this discards:** any residual phase-angle error in the total phase phasor. The assumption is that after two passes the phase angle is already correct.

```python
currentVector = np.concatenate(currentsPh)
resultsCurrentVector = np.abs(currentVector)    # magnitude |I_i| for each cell
```

The final reported quantity is the RMS magnitude of the complex current in each cell.

---

## 7. Power losses post-processing

### Resistance vector

```python
resistanceVector = csdm.getResistanceArray(
    elementsVector,
    dXmm=dXmm, dYmm=dYmm,
    temperature=temperature,
    lenght=length,
    sigma_array=sigma_array,     # shape (N,) — per-element σ
    alpha_array=alpha_array,     # shape (N,) — per-element α
)
```

Returns a shape-(N,) array of resistances in Ω (for the given conductor length). Each element has the temperature-corrected resistance of its material.

### Losses per element

```python
powerLossesVector = resistanceVector * resultsCurrentVector**2   # shape (N,) [W]
```

$P_i = R_i \cdot |I_i|^2$

Note: `resultsCurrentVector` contains RMS magnitudes (not instantaneous peak values), so this directly gives ohmic power in watts (for the analysed length).

### Total and per-phase losses

```python
powerLosses = np.sum(powerLossesVector)      # scalar [W]

powPh = []
start = 0
for n_elem in elementsPhase:
    powPh.append(np.sum(powerLossesVector[start : start + n_elem]))
    start += n_elem
```

Because `elementsVector` is ordered phase-by-phase, slicing by `elementsPhase` counts directly maps to phases.

### Return value of `solve_with_magnetic`

```python
return (
    resultsCurrentVector,          # shape (N,)  — |I_i| per cell [A]
    (powerLosses, powPh),          # scalar + list — total [W] and per-phase [W]
    elementsVector,                # shape (N,5)  — cell metadata
    powerLossesVector,             # shape (N,)   — P_i per cell [W]
    currentVector,                 # shape (N,)   — complex I_i per cell [A]
    vPh,                           # list of per-phase element arrays
    mi_r_weighted_array            # shape (rows,cols) — μr_w map
)
```

---

## 8. Result array reconstruction

To display results on the 2-D cross-section grid:

```python
def recreateresultsArray(elementsVector, resultsVector, initialGeometryArray, dtype=float):
    localResultsArray = np.zeros(initialGeometryArray.shape, dtype=dtype)
    rows = elementsVector[:, 0].astype(int)
    cols = elementsVector[:, 1].astype(int)
    localResultsArray[rows, cols] = resultsVector
    return localResultsArray
```

This scatter-assigns the 1-D result vector back onto the 2-D grid using the row/col indices stored in `elementsVector`. Empty cells remain 0.

Called twice in typical post-processing:
```python
resultsArray      = recreateresultsArray(elementsVector, resultsCurrentVector, XSecArray)
resultsArrayPower = recreateresultsArray(elementsVector, powerLossesVector, XSecArray)
```

---

## 9. Thermal solver (bar-level)

`solve_thermal_for_bars` (`csdsolve.py:721`) uses a conductance-matrix approach for steady-state temperature rise of individual conductor bars.

### Physical model

Each bar (physical conductor) is characterised by:
- `bar.power` — ohmic heat generation [W]
- `bar.perymiter` — surface perimeter [mm]
- `bar.length` — axial length [mm]
- `bar.Rth` — end-to-end longitudinal thermal resistance [K/W]
- `bar.thermal_group` — bars in the same group are thermally connected at both ends

Heat paths modelled:
1. **Surface convection** — heat from the bar surface to ambient air
2. **End conduction** — longitudinal heat flow along the conductor to bars in the same phase/group

### Conductance matrix assembly

```python
B = len(list_of_bars)
thermal_G_matrix_cond = np.zeros((B, B))   # bar-to-bar conduction
thermal_G_matrix      = np.zeros((B, B))   # full system matrix
vector_Q              = np.zeros(B)         # heat sources

# Off-diagonal: bar-to-bar longitudinal conduction
for r, bar_n in enumerate(list_of_bars):
    for c, bar_m in enumerate(list_of_bars):
        same_group = (bar_n.thermal_group != 0 and bar_n.thermal_group == bar_m.thermal_group)
        same_phase = (bar_n.thermal_group == 0 and bar_n.phase == bar_m.phase)
        if (same_group or same_phase) and bar_n is not bar_m:
            thermal_G_matrix_cond[r, c] = 2 / (bar_n.Rth + bar_m.Rth)

# System matrix assembly
for r, bar_n in enumerate(list_of_bars):
    for c, bar_m in enumerate(list_of_bars):
        if bar_n is bar_m:
            G_surface = bar_n.length * bar_n.perymiter * 1e-6 * HTC
            thermal_G_matrix[r, c] = G_surface + thermal_G_matrix_cond[r, :].sum()
            vector_Q[r] = bar_n.power
        else:
            thermal_G_matrix[r, c] = -thermal_G_matrix_cond[r, c]
```

**Diagonal (bar n):**
$$G_{nn} = \underbrace{h_{tc} \cdot l \cdot P_{bar}}_{\text{surface convection}} + \sum_{m \neq n} G_{nm,\text{cond}}$$

where $h_{tc}$ is the heat-transfer coefficient [W/(m²·K)], $l$ is length in m, and $P_{bar}$ is perimeter in m (converted from mm via `× 1e-6 m²/mm²` which handles the mm² → m² conversion for the product `length[mm] × perimeter[mm]`).

**Off-diagonal:**
$$G_{nm} = -G_{nm,\text{cond}} = -\frac{2}{R_{th,n} + R_{th,m}}$$

The factor of 2 accounts for thermal connection at *both* ends of the bar pair.

### Solve

```python
inverse_G_matrix = np.linalg.inv(thermal_G_matrix)
dT_vector = np.matmul(inverse_G_matrix, vector_Q)

for bar, dt in zip(list_of_bars, dT_vector):
    bar.dT = dt
```

$$[\mathbf{G}_{th}] \cdot [\Delta T] = [Q]$$
$$[\Delta T] = [\mathbf{G}_{th}]^{-1} \cdot [Q]$$

`bar.dT` is the steady-state temperature rise of that bar above ambient [K].

---

## 10. Conductor detection

`getConductors` (`csdfunctions.py:189`) identifies individual physically separate conductors within each phase (e.g. the six bars of a double-busbar system, or parallel cables).

```python
def getConductors(XsecArr, phases):
    conductorsArr = np.zeros(XsecArr.shape, dtype=int)
    conductors_number = 0
    phaseCond = []
```

**Algorithm:**

1. Mark all cells of each phase with a negative sentinel value (`-1 - phase_index`) to distinguish them from assigned conductors.
2. Scan the array row by row, column by column.
3. For each unmarked cell of a phase, look at its 8 neighbours plus the same row up to 5 columns ahead.
4. If a neighbour already has a positive conductor number, adopt that number (connectivity propagation).
5. If no numbered neighbour is found, assign a new conductor number.

The result is `conductorsArr`: same shape as `XSecArray`, each non-zero cell holds the integer index of the physical conductor it belongs to.

```python
return conductorsArr, conductors_number, phaseCond
```

- `conductorsArr` — 2-D array of conductor IDs
- `conductors_number` — total count of detected conductors
- `phaseCond` — list of lists: `phaseCond[n]` = conductor IDs belonging to phase n

**Limitation:** the look-ahead heuristic (5 columns) can miss thin connections and sometimes splits a single conductor into two. Works reliably for rectangular busbar geometry; less reliable for complex shapes.

---

## 11. Solver variants

Three solver functions exist in `csdsolve.py`. They share the same mathematical structure but differ in flexibility:

| Function | Phases | Materials | Magnetic μr | Used by |
|---|---|---|---|---|
| `solve_system` | Exactly 3 (hardcoded A/B/C) | Single global σ, α | No | Legacy code only |
| `solve_multi_system` | N (general) | Per-phase σ, α | No | — |
| `solve_with_magnetic` | N (general) | Per-phase σ, α, μr, μr_w | Yes | CLI, Pro solver |

`solve_with_magnetic` is the current production solver. Its additional cost over `solve_multi_system` is the `get_mi_weighted` pre-computation, which is O(N_cells × delta_cells²) — potentially the dominant cost for large geometries with large delta values.

`solve_gui_3f` is a thin wrapper around `solve_multi_system` that takes the old scalar-parameter interface of the GUI and converts it to the multi-phase call.

---

## 12. Complexity and performance notes

| Step | Cost | Notes |
|---|---|---|
| `arraySlicer` (×2) | O(rows × cols) | Quadruples array size |
| `arrayVectorize` | O(N) | argwhere + column_stack |
| `get_mi_weighted` | O(N_cells × delta_cells²) | Expensive for large delta; default delta=250mm |
| `getDistancesArray` | O(N²) | Broadcasting, no loops; fits in memory up to N~10000 |
| `getMutualInductance` (all pairs) | O(N²) | Vectorised; dominates Z assembly |
| `getGmatrix` = `np.linalg.inv` | **O(N³)** | **Dominant cost for most cases** |
| First and second solve (`matmul`) | O(N²) | Cheap once G is known |
| `getResistanceArray` | O(N) | Element-wise |
| `powerLossesVector` | O(N) | Element-wise |

### Memory

The impedance/admittance matrix is `N × N × 16 bytes` (complex128):

| N cells | Memory |
|---|---|
| 500 | 4 MB |
| 1000 | 16 MB |
| 2000 | 64 MB |
| 5000 | 400 MB |

### Numba

`getSelfInductance`, `getMutualInductance`, `getResistance`, `getResistanceArray`, `getPerymiter`, `getDistancesArray`, `get_mi_weighted`, and `get_mi_averaged` are decorated with `@conditional_decorator(njit, use_njit)`. If Numba is installed, these are JIT-compiled to native code on the first call. The warm-up is ~1–2 s; subsequent calls are typically 5–20× faster.

`getImpedanceArray` itself is not JIT-compiled (the comment shows it was intentionally disabled) because it relies on `np.fill_diagonal` which Numba does not fully support. The inner functions it calls (`getMutualInductance`, `getSelfInductance`, `getResistance`) are JIT-compiled when available.

### The two-pass vs iterative approach

The current implementation does exactly **two** G·U multiplications per solve. A true iterative solver would repeat until convergence. For the physics encountered here (busbars, cables), one voltage correction is sufficient because:

1. The system is linear — the ratio `I_computed / U_trial` gives the exact effective admittance.
2. The voltage correction is therefore exact in one step for the total phase currents.
3. The residual error (handled by step 6.6 normalisation) comes only from floating-point arithmetic, not physical non-linearity.

The magnitude normalisation in step 6.6 is a final trim to machine-precision correctness, not a convergence iteration.
