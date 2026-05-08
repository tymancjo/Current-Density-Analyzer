# Current Density Analyzer — User Guide

**CSD2** is an open-source tool for analysing the two-dimensional distribution of AC current density in arbitrary conductor cross-sections. It computes skin and proximity effects, ohmic power losses, impedance parameters, and electrodynamic forces for conductor arrangements of any shape and material.

The tool is available in three forms that share the same physics engine:

| Interface | File | Use case |
|---|---|---|
| Graphical UI | `csd2.py` | Interactive design and analysis |
| Command-line | `cliCSD.py` | Scripting, batch runs, CI |
| Geometry editor | `csdeditor/editor.html` | Visual `.ic` file creation |

---

## Table of Contents

1. [Physics and method](#1-physics-and-method)
2. [Installation](#2-installation)
3. [The `.ic` geometry file](#3-the-ic-geometry-file)
4. [Using the graphical UI (csd2.py)](#4-using-the-graphical-ui-csd2py)
5. [Using the command-line interface](#5-using-the-command-line-interface)
6. [Using the geometry editor](#6-using-the-geometry-editor)
7. [Materials](#7-materials)
8. [Interpreting results](#8-interpreting-results)
9. [Accuracy and performance](#9-accuracy-and-performance)
10. [Limitations](#10-limitations)

---

## 1. Physics and method

### 1.1 What the tool models

At power-frequency AC, two effects cause the current density inside a conductor to be non-uniform:

- **Skin effect** — the current is pushed toward the conductor surface by its own changing magnetic field.
- **Proximity effect** — nearby conductors carrying other (or opposite) currents distort the distribution further.

Both effects intensify with frequency, conductor size, and conductivity. CSD computes the full spatial distribution of current density across the cross-section, accounting for both effects simultaneously and for any arrangement of conductors.

### 1.2 Discretisation

The cross-section is approximated by a uniform rectangular grid of small square cells. Each cell is treated as an independent current-carrying filament (a "wire") of length equal to the analysed conductor length. Within each cell the current density is assumed uniform.

Choosing a smaller cell size (finer mesh) increases accuracy at the cost of more computation. A cell size of **1–2 mm** is a good balance for most busbar work; cables and smaller conductors may need 0.5 mm or less.

### 1.3 Impedance matrix

For a system of *N* elemental wires, the voltage–current relationship is:

$$\mathbf{U} = \mathbf{Z} \cdot \mathbf{I}$$

The N × N impedance matrix **Z** is assembled as follows:

**Diagonal terms** — self-impedance of wire *i*:

$$Z_{ii} = R_i + j\omega L_i$$

where the resistance is:

$$R_i = \frac{l}{\sigma(T) \cdot A}$$

with $\sigma(T) = \sigma_{20°C} \cdot \frac{1}{1 + \alpha(T - 20)}$, *l* the conductor length, and *A* the cell cross-section area.

The self-inductance of a rectangular wire is (Neumann formula for finite-length conductors):

$$L_i = \frac{\mu_0 \mu_r}{2\pi} \left( \ln\frac{2l}{r} - 1 + \frac{\mu_r}{4} \right) \cdot l$$

where *r* is the equivalent radius of the cell cross-section.

**Off-diagonal terms** — mutual inductance between wires *i* and *j*:

$$Z_{ij} = j\omega M_{ij}$$

$$M_{ij} = \frac{\mu_0 \mu_{r,\text{medium}}}{2\pi} \cdot l \left( \ln \frac{l + \sqrt{l^2 + d^2}}{d} - \sqrt{1 + \left(\frac{d}{l}\right)^2} + \frac{d}{l} \right)$$

where *d* is the distance between the centres of wires *i* and *j*.

### 1.4 Solution algorithm

The tool is given the **RMS current magnitude and phase angle** for each conductor group (phase), not a voltage. Solving requires a two-pass process:

1. **Build Z** — assemble the full N × N impedance matrix from cell geometry and material properties.
2. **Invert Z** — compute the admittance matrix **G** = **Z**⁻¹. This is the computationally expensive step; its cost scales as O(N³).
3. **First solve** — assign a trial voltage vector **U₀** with unit magnitude and the correct phase angles (0°, 120°, 240° for balanced three-phase). Compute **I₀** = **G** · **U₀** and sum the currents by phase.
4. **Voltage correction** — scale each phase voltage by the ratio of the target current to the computed current:

$$U_\phi \leftarrow U_\phi \cdot \frac{I_{\phi,\text{target}}}{I_{\phi,\text{computed}}}$$

5. **Second solve** — recompute **I** = **G** · **U** with the corrected voltages. A final normalisation step brings each phase current exactly to the target RMS value.
6. **Power losses** — for each cell: $P_i = R_i \cdot |I_i|^2$. Phase and total losses are the respective sums.

A single correction pass converges well for typical balanced arrangements. The admittance matrix is only computed once, so the correction step is essentially free.

### 1.5 Magnetic materials

When the geometry includes materials with relative permeability μᵣ > 1 (e.g. carbon steel enclosures), the effective permeability experienced by each wire is estimated as a weighted average of all surrounding materials, with weights inversely proportional to distance. This allows the tool to capture how paramagnetic elements concentrate or divert flux, which in turn shifts the current distribution.

---

## 2. Installation

### Requirements

| Package | Minimum version | Purpose |
|---|---|---|
| Python | 3.9 | Runtime |
| NumPy | 1.21 | Matrix operations |
| matplotlib | 3.4 | Result plots |
| CustomTkinter | 5.x | Modern UI widgets |
| Numba | 0.56 (optional) | JIT acceleration of hot loops |

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourorg/Current-Density-Analyzer
cd Current-Density-Analyzer

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install numpy matplotlib customtkinter
pip install numba            # optional — speeds up large geometries

# 4. Launch the UI
python csd2.py
```

The geometry editor (`csdeditor/editor.html`) is a self-contained single HTML file — open it directly in any modern browser. No server or installation needed.

---

## 3. The `.ic` geometry file

A `.ic` file is a plain-text script that fully describes a cross-section. It defines:

- the **geometry** (which cells belong to which phase/conductor)
- the **material** of each conductor
- the **electrical excitation** (RMS current and phase angle) of each conductor

`.ic` files are the common input format shared by the UI, the CLI, and the geometry editor.

**Full reference:** `doc/innercode_doc.md`

### Quick overview

```
# Comments start with #

# Variables
v(I,  2300)         # define a variable: I = 2300 A
v(w,  10)           # bar width = 10 mm
v(h,  60)           # bar height = 60 mm
v(p,  1)            # starting phase ID

# Material assignments (ID from setup/materials.txt)
material(1, 2)      # phase 1 → Aluminium
material(2, 2)
material(3, 2)

# Electrical excitation
current(1, I,   0, 0)    # phase 1: I A at   0°
current(2, I, 120, 0)    # phase 2: I A at 120°
current(3, I, 240, 0)    # phase 3: I A at 240°

# Geometry — r(x, y, width, height, phase_id)
v(x, 0)
l(3)
    r(x, 0, w, h, p)
    a(x, 20)    # x += 20
    a(p,  1)    # next phase ID
break
```

### Geometry commands

| Command | Description |
|---|---|
| `r(x, y, w, h, phase)` | Filled rectangle; (x,y) is bottom-left corner |
| `c(x, y, D, phase)` | Solid disc; (x,y) is centre, D is diameter |
| `c(x, y, D_out, phase_out, D_in, phase_in)` | Annulus (hollow cable) |
| `mv(phase, dx, dy)` | Move all cells of a phase |
| `cp(phase, dx, dy)` | Copy all cells of a phase |

All dimensions are in **millimetres**. The coordinate origin (0, 0) is the bottom-left of the canvas.

### Loops

```
l(n)
    ... body executed n times ...
break
```

Variables changed inside the loop carry forward to the next iteration. Loops can be nested.

### Expressions

Variable values and command arguments can be any arithmetic expression using `+  -  *  /  **`, together with `sin`, `cos`, `sqrt`, `pi`, `log`, `abs`, `min`, `max`, and the full Python `math` module.

```
c(r*cos(angle), r*sin(angle), D, ph)   # polar-to-Cartesian placement
```

---

## 4. Using the graphical UI (csd2.py)

Start the application:

```bash
python csd2.py
```

### 4.1 Main window layout

The window is divided into four areas:

```
┌─────────────────────────────────────────┐
│  Menu bar                               │
├───────────────────┬─────────────────────┤
│                   │  Controls panel     │
│  Drawing canvas   │  (right side)       │
│                   │                     │
├───────────────────┴─────────────────────┤
│  Inner-code (.ic) text editor           │
└─────────────────────────────────────────┘
```

### 4.2 Drawing geometry

**Direct drawing** on the canvas:

- Select a phase colour from the toolbar (L1 red, L2 green, L3 blue, or custom).
- Left-click and drag to paint cells of the selected phase.
- Right-click or select the eraser to clear cells.
- Use the zoom and pan controls to navigate.

**Via `.ic` code** (recommended for parametric work):

1. Type or paste the `.ic` code in the text editor at the bottom.
2. Click **Get Canvas** to parse and render the geometry at the optimal cell size.
3. Click **InterCode** to execute the code and paint the cells on the current canvas.

The IC code editor supports loading files: `File → Open .ic`.

### 4.3 Setting analysis parameters

In the controls panel (right side) set:

| Parameter | Description | Typical value |
|---|---|---|
| Cell size (mm) | Grid resolution | 1–5 mm |
| Frequency (Hz) | Supply frequency | 50 or 60 Hz |
| Current (A) | Phase RMS current | depends on application |
| Temperature (°C) | Conductor temperature | 20–150 °C |
| Length (mm) | Analysed conductor length | 1000 mm (1 m) |

### 4.4 Available analyses

All analyses are accessible from the **Analyze** menu:

| Analysis | Description |
|---|---|
| **Current Density** | Compute and display the current density distribution across the cross-section |
| **Power Losses** | Quick ohmic loss calculation; results shown in the console |
| **Power Losses ProSolver** | Full multi-phase solver with per-phase material selection and thermal output |
| **Impedances** | Compute self and mutual impedances per phase |
| **Forces** | Electrodynamic force vectors between conductors (static case) |
| **Thermal** | Simplified thermal analysis (steady-state temperature rise) |

### 4.5 The Pro Power Losses Solver

The Pro solver window accepts fine-grained input per phase:

- **Current RMS [A; deg]** — enter as a semicolon-separated list: `Irms₁;φ₁;Irms₂;φ₂;…`  
  Example for balanced 3-phase: `2300;0;2300;120;2300;240`
- **Frequency** and **length** as above.
- **Material tab** — assign a material from `setup/materials.txt` independently to each phase.
- **Thermal tab** — set ambient temperature, heat-transfer coefficient, and conductor surface area for a steady-state thermal calculation.

When the geometry was loaded from an `.ic` file that contains `current()` and `material()` declarations, the Pro solver pre-fills these values automatically.

Click **Calculate!** to run the solver, then **Show Results** to display the current density and power loss maps.

### 4.6 Saving and loading

| Operation | How |
|---|---|
| Save geometry | `File → Save` — stores the NumPy array as a `.csd` file |
| Load geometry | `File → Open` — loads a `.csd` file |
| Load `.ic` | `File → Open .ic` — loads an inner-code file into the editor |
| Export image | Results plots can be saved via the matplotlib toolbar |

---

## 5. Using the command-line interface

`cliCSD.py` runs a complete analysis from the terminal, reading geometry from an `.ic` file and writing results to stdout.

### Basic usage

```bash
python cliCSD.py [options] geometry.ic current
```

`geometry.ic` — path to the `.ic` geometry file  
`current` — RMS current in amperes (used as fallback if `.ic` has no `current()` lines)

### Options reference

| Flag | Long form | Default | Description |
|---|---|---|---|
| `-s N` | `--size` | 5 | Maximum cell size in mm; geometry is subdivided until cells are ≤ N mm |
| `-f N` | `--frequency` | 50 | Frequency in Hz |
| `-T N` | `--temperature` | 140 | Conductor temperature in °C |
| `-l N` | `--length` | 1000 | Analysed length in mm |
| `-htc N` | `--htc` | 5 | Heat-transfer coefficient W/(m·K) |
| `-sig N` | `--conductivity` | 56×10⁶ | Conductivity at 20 °C in S/m |
| `-rco N` | `--temRcoeff` | 3.9×10⁻³ | Temperature coefficient of resistance 1/K |
| `-mat N` | `--material` | −1 | Material index from `setup/materials.txt` (overrides default copper) |
| `-sp` | `--simple` | off | Compact output (single line per frequency) |
| `-md` | `--markdown` | off | Per-conductor results as a Markdown table |
| `-csv` | `--csv` | off | CSV output: `frequency,total_losses` |
| `-v` | `--verbose` | off | Detailed solver log |
| `-d` | `--draw` | off | Pop up a geometry preview window before solving |
| `-r` | `--results` | off | Pop up a results window after solving |
| `-b` | `--bars` | off | Detect individual conductors and report per-conductor losses |
| `-bd` | `--bardetails` | off | Show per-conductor result on the plot (requires `-b`) |

### Examples

```bash
# Single analysis — 3-phase busbars at 2300 A, 50 Hz
python cliCSD.py input/example.ic 2300

# Finer mesh, 60 Hz, verbose output
python cliCSD.py -s 2 -f 60 -v input/example.ic 2300

# Sweep — power losses at 10 frequencies (bash loop)
for f in 10 20 30 40 50 60 100 150 200 400; do
  python cliCSD.py -f $f -csv input/example.ic 2300
done

# Show geometry and results graphically
python cliCSD.py -d -r input/example.ic 2300

# Per-bar reporting as markdown table
python cliCSD.py -b -md input/example.ic 2300

# Use aluminium material (ID 2) from materials database
python cliCSD.py -mat 2 input/example.ic 2300
```

### Current and material from `.ic` file

If the `.ic` file contains `current()` declarations, those values take priority over the positional `current` argument on the command line. The CLI argument then serves only as a fallback for geometries without `current()` lines.

```
# Inside the .ic file:
current(1, 2300, 0,   0)
current(2, 2300, 120, 0)
current(3, 2300, 240, 0)
```

Similarly, `material()` declarations in the `.ic` file map each phase to a specific entry in `setup/materials.txt`.

### Output format

Default output (example):

```
Starting operations...
dX:2.5mm dY:2.5mm
Data table size: (48, 12)

--- Results ---
Total power losses: 32.47 W/m
Phase 1: 10.82 W/m
Phase 2: 10.83 W/m
Phase 3: 10.82 W/m
```

With `-b` (bar detection):

```
Conductor  Phase  Area[mm²]   I[A]    P[W/m]
    1         1     600.0    2300.0    10.82
    2         2     600.0    2300.0    10.83
    3         3     600.0    2300.0    10.82
```

---

## 6. Using the geometry editor

The geometry editor is a standalone HTML5 application. Open `csdeditor/editor.html` in any modern browser — no installation or server required.

### 6.1 Interface overview

```
┌────────────────── Toolbar ──────────────────────────┐
│ ▲ Select  ▭ Rect  ● Circle │ Snap │ Fit │ ⤵ Import │
├─── Phases ──┬─────── Canvas ──────┬── Properties ───┤
│ L1  ●  #1  │                     │ Shape position   │
│ L2  ●  #2  │   Drawing area      │ Phase settings   │
│ L3  ●  #3  │   (mm coordinates)  │ Material / I rms │
│             │                     │                  │
│ + Add Phase │                     │                  │
├─────────────┴─────────────────────┴──────────────────┤
│ .ic Code   [↑ Load]  [⎘ Copy]  [⬇ Download .ic]      │
│ # Generated .ic code appears / can be pasted here    │
└──────────────────────────────────────────────────────┘
```

### 6.2 Drawing shapes

**Tools** (toolbar or keyboard shortcut):

| Tool | Key | Action |
|---|---|---|
| Select | `V` | Click to select; drag to move |
| Rectangle | `R` | Click and drag to draw a rectangle |
| Circle | `C` | Click and drag from centre outward |
| Fit view | `F` | Zoom and pan to show all shapes |
| Escape | `Esc` | Return to Select, clear selection |

The active phase (highlighted in the left panel) is assigned to new shapes.

**Snap** — round all coordinates to the nearest N mm (default 5 mm). Adjust in the toolbar dropdown. Set to "None" for free placement.

### 6.3 Navigation

| Action | How |
|---|---|
| Zoom | Mouse wheel |
| Pan | Middle-click drag, or Space + drag |
| Fit all content | `F` key or Fit button |

Coordinates are always displayed in mm in the toolbar and the canvas overlay.

### 6.4 Selecting and editing shapes

- **Single click** — select one shape; its properties appear in the right panel.
- **Ctrl+click** — add / remove a shape from the selection.
- **Drag on empty canvas** — rubber-band marquee selects all shapes it touches.
- **Ctrl+A** — select all.

With a shape selected, edit its exact position, width/height (rect) or diameter (circle) in the **Properties** panel. You can also change its phase and, for the circle tool, define an inner diameter to create an annulus.

When a phase is selected in the properties, its current (Irms, phase angle) and material are shown and editable directly there.

### 6.5 Shape operations

| Operation | Shortcut |
|---|---|
| Delete selected | `Del` |
| Duplicate | `Ctrl+D` |
| Copy | `Ctrl+C` |
| Paste (with offset) | `Ctrl+V` |
| Assign phase to all selected | Multi-select → "Set Phase for All" dropdown |

### 6.6 Phase management

The left panel lists all defined phases. Each phase has:
- A colour swatch (click to change with a colour picker).
- A label (editable in the Properties panel).
- A numeric ID (used in the generated `.ic` code).
- Per-phase current (Irms and angle) and material settings.

Click **+ Add Phase** to create a new one. Delete a phase with the × button on its row (you will be warned if shapes are using it).

### 6.7 Exporting to `.ic`

The bottom panel shows a live preview of the generated `.ic` code. It updates every time you draw, move, or change a property.

- **Copy** — copy the code to the clipboard.
- **Download .ic** — save the file directly.

The code is flat (no loops), suitable for direct use in the UI or CLI, and can be re-imported into the editor (see below).

### 6.8 Importing `.ic` files

Three ways to load an existing `.ic` file:

1. **Import button** — click ⤵ Import in the toolbar and choose a file.
2. **Drag and drop** — drag a `.ic` file onto the canvas.
3. **Paste and load** — paste `.ic` code directly into the bottom textarea, then click **↑ Load** (or press `Ctrl+Enter` inside the textarea).

> **Note:** The import parser handles flat `.ic` files (the format the editor itself generates). Files with `l(n)…break` loops are not supported for import — use the CLI or UI to run those files, then paste the generated geometry output back if needed.

---

## 7. Materials

Materials are defined in `setup/materials.txt`. The built-in database:

| ID | Name | σ at 20°C [S/m] | α [1/K] | μᵣ | Typical use |
|---|---|---|---|---|---|
| 0 | Copper | 56×10⁶ | 3.9×10⁻³ | 1 | Busbars, cable conductors |
| 1 | FakeAl | 31×10⁶ | 4.4×10⁻³ | 1 | Placeholder for aluminium |
| 2 | Aluminium | 31×10⁶ | 4.4×10⁻³ | 1 | Al conductors |
| 3 | Carbon Steel | 6.99×10⁶ | 6.5×10⁻³ | 100 | Structural steel, frames |
| 4 | Stainless Steel | 1.45×10⁶ | 1.0×10⁻³ | 1 | Non-magnetic structural steel |
| 5 | Fake Carbon Steel | 6.99×10⁶ | 6.5×10⁻³ | 10 | Short-section approximation |

The file is semicolon-delimited and can be extended with custom materials by appending rows in the same format:

```
name; conductivity [S/m]; alpha [1/K]; density [kg/m³]; cp [J/kg·K]; mi_r [-]; thermal_cond [W/m·K]
```

---

## 8. Interpreting results

### Current density map

The colour-mapped canvas shows the current density in each cell. Higher density regions (hot colours) indicate where the skin or proximity effect concentrates current. In a single round conductor the skin effect alone would show a ring-shaped distribution; proximity from nearby conductors produces asymmetric distortion.

The **AC/DC current density ratio** (sometimes called the *current distribution factor*) integrated over the cross-section must equal 1 — the solver enforces the correct total current.

### Power losses

Reported as **W per metre of conductor length**. The total figure accounts for the non-uniform distribution; it will be higher than the DC value (calculated from bulk resistance) by the factor:

$$k_{ac} = \frac{P_{ac}}{P_{dc}} = \frac{\sum_i R_i |I_i|^2}{R_{bulk} \cdot I_{rms}^2} \geq 1$$

This factor is what the tool is most often used to compute — it feeds directly into cable ampacity or busbar temperature-rise calculations.

### Per-conductor breakdown (bars)

With the `-b` flag (CLI) or the bar-detection feature (UI), the solver identifies individual physical conductors within each phase and reports the current and losses per conductor. This is useful for parallel bar arrangements where current sharing is uneven.

### Temperature rise (thermal solver)

The thermal module uses the computed power loss density (W/m²) as a heat source and applies Newton's law of cooling at the conductor surface:

$$\Delta T = \frac{P_{losses}}{h_{tc} \cdot A_{surface}}$$

where $h_{tc}$ is the heat-transfer coefficient (W/m²·K, entered in the Pro solver Thermal tab) and $A_{surface}$ is the conductor perimeter per unit length. The result is the steady-state temperature rise above ambient.

---

## 9. Accuracy and performance

### Cell size guidelines

| Cell size | Typical use | Notes |
|---|---|---|
| 5–10 mm | Quick estimates, large busbars | Fast; lower accuracy for skin effect |
| 2–3 mm | General busbar work | Good balance |
| 1–1.5 mm | Accurate skin/proximity analysis | Recommended for publication results |
| < 1 mm | High-precision, small conductors | May be slow; > 1000 cells triggers a warning |

The solver's computational cost is **O(N³)** in the number of cells. Doubling the mesh refinement (halving cell size) multiplies the computation time by roughly 8×.

### Numba acceleration

If `numba` is installed, hot loops (distance array computation, inductance formulas) are JIT-compiled to native code. This typically reduces solve time by 3–10× for larger meshes. Install with:

```bash
pip install numba
```

### Element count

| Cell count | Expected behaviour |
|---|---|
| < 400 | Near-instant |
| 400–1200 | Seconds |
| 1200–5000 | Minutes; a warning is printed |
| > 10000 | Very long; "extreme size" warning is printed |

Use the cell-size slider or the `-s` option to control mesh density.

---

## 10. Limitations

- **2D cross-section only.** End effects, bends, and joints are not modelled. The analysed length is assumed uniform.
- **Sinusoidal (AC) currents only.** The solver works in the frequency domain. Non-sinusoidal waveforms (PWM, harmonics) require separate runs at each harmonic frequency and superposition of losses.
- **Magnetostatics for forces.** The force solver assumes constant (DC) current. Force results under AC conditions represent the time-averaged (mean) electrodynamic force, not the peak.
- **No radiation heat transfer.** The thermal solver uses convection only (Newton cooling). Radiation can be significant at high temperatures.
- **No eddy-current heating in magnetic enclosures.** The magnetic material model adjusts inductances but does not separately solve for eddy-current losses in steel frames.
- **Import of looped `.ic` files into the editor.** The HTML editor parses flat `.ic` code only. Files using `l(n)…break` must be run through the Python parser first.

---

## Quick-start workflow

1. **Sketch the geometry** — use the HTML editor to draw bars or cables visually, assign phases, set currents and materials, download the `.ic` file.
2. **Quick check** — load the `.ic` file in the CLI for a fast result at default settings:
   ```bash
   python cliCSD.py -b input/mycase.ic 1000
   ```
3. **Refine** — open the file in the UI (`csd2.py`), tune cell size, run the Pro solver for thermal output.
4. **Iterate** — edit the `.ic` file (or re-open the editor), adjust geometry, re-run.

---

*For `.ic` file syntax details see `doc/innercode_doc.md`.  
For the underlying theory see `doc/working_idea.md`.*
