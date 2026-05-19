"""
This file is intended to be the Command Line Interface
fot the CSD tool aimed to the quick analysis
for power losses in given geometry.
The idea is to be able to use the saved geometry file
and deliver the required input as a command line
parameters.

As an output the csdf.myLoged info of power losses
is generated on the standard output.
"""

# TODO:
# 1. Read the command line parameters - done
# 2. Loading the main geometry array from the file - done
# 3. Setup the solver - done
# 4. Solve - done
# 5. Prepare results - done
# 6. csdf.myLog results - done
# 7. adding inner code working - done
# 8. clean and make use of modules - done
# 9. adding support of the materials - by the line parameters. done
# 10. Use of material database file - done
# 11. Use multiphase currents definition - by a currentfile? - done by the 'current(phase, Irms, elshift, extra shift)' innerocde params
# 12. Use multiple materials - define material per phase by a innercode - DONE
# 13. Clean up the phases numbering - allow for any numbers - DONE
# 14. Add the MoveToPhase(A->B) command to the IC. 
# 99. Update the doc.


# General imports
import numpy as np

# import os.path
import sys

# import pickle
import argparse

# Importing local library
from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm
from csdlib import csdsolve as csds
from csdlib import csdos


def getArgs():
    """
    Handling the cli line parameters.
    """
    parser = argparse.ArgumentParser(
        description="CSD cli executor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--size", help="Max single cell size in [mm] (overrides .ic cellsize)", type=float, default=None
    )
    parser.add_argument(
        "-f", "--frequency", type=float, default=None, help="Currents frequency in Hz (overrides .ic freq)"
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=None,
        help="Conductors temperature in deg C (overrides .ic temp)",
    )
    parser.add_argument(
        "-l", "--length", type=float, default=None, help="Analyzed length (overrides .ic length)"
    )
    parser.add_argument(
        "-htc",
        "--htc",
        type=float,
        default=None,
        help="Heat transfer coefficient for cooling of conductors in [W/m.K] (overrides .ic htc)",
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
        parser.add_argument(
            "-F",
            "--forces",
            action="store_true",
            help="Compute electromagnetic forces on each conductor (implies --bars). "
                 "Isc values are read from .ic file current() 5th arg, or set via --isc.",
        ),
        parser.add_argument(
            "--isc",
            nargs="+",
            type=float,
            default=None,
            metavar="Isc_kA",
            help="Short-circuit currents [kA] per phase in order (signed). "
                 "Overrides .ic file Isc values. E.g.: --isc 30 -15 -15",
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


def main():
    """
    This is the place where the main flow of operation is carried.
    """

    config = getArgs()
    verbose = config["verbose"]
    simple = config["simple"]
    csv = config["csv"]

    # Forces analysis requires bar detection
    if config["forces"]:
        config["bars"] = True

    # for simplicity so the log procedure can see it globally
    csdf.verbose = verbose

    csdf.myLog()
    csdf.myLog("Starting operations...")
    csdf.myLog()

    if config["draw"] or config["results"]:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm

    XSecArray = np.zeros((0, 0))
    dXmm = dYmm = 1

    # 2 loading the geometry and other data:
    XSecArray, dXmm, dYmm, currents, materials, custom_materials, analysis_params = \
        csdf.loadTheData(config["geometry"])

    # Apply analysis_params from .ic file for any CLI arg that wasn't explicitly set
    config["size"]        = config["size"]        or analysis_params.get("cellsize", 5.0)
    config["frequency"]   = config["frequency"]   or analysis_params.get("freq",     50.0)
    config["temperature"] = config["temperature"] or analysis_params.get("temp",     140.0)
    config["length"]      = config["length"]      or analysis_params.get("length",   1000.0)
    config["htc"]         = config["htc"]         or analysis_params.get("htc",      5.0)

    csdf.myLog("Initial geometry array parameters:")
    csdf.myLog(f"dX:{dXmm}mm dY:{dYmm}mm")
    csdf.myLog(f"Data table size: {XSecArray.shape}")
    csdf.myLog(f"Currents definition: {currents}")
    csdf.myLog(f"Material definition pattern: {materials}")
    if analysis_params:
        csdf.myLog(f"Analysis params from .ic: {analysis_params}")

    list_of_phases = np.unique(XSecArray).astype(int)
    list_of_phases = [int(n) for n in list_of_phases if n != 0]
    original_phase_index = {index: phase for index, phase in enumerate(list_of_phases)} 
    new_phase_index = {phase: index for index, phase in enumerate(list_of_phases)} 
    # normalizing the phases numbering
    normalized_XsecArr = np.zeros(XSecArray.shape)
    for index, phase in enumerate(list_of_phases, start=1):
        normalized_XsecArr[XSecArray == phase] = index
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
        for n in range(number_of_phases):
            Icurrent.append((Irms, phi[x] + direction[x]))
            x += 1
            if x >= len(phi):
                x = 0

    # Build Isc_per_phase dict: keyed by original phase ID, value in kA (signed)
    # Priority: CLI --isc > .ic file 5th current() arg > 0
    Isc_per_phase = {}
    if len(currents) == number_of_phases:
        for c in currents:
            if len(c) >= 5:
                Isc_per_phase[int(c[0])] = float(c[4])
    if config.get("isc"):
        # CLI override: values in kA, ordered by phase (list_of_phases order)
        for ph_idx, isc_val in enumerate(config["isc"]):
            if ph_idx < len(list_of_phases):
                Isc_per_phase[list_of_phases[ph_idx]] = float(isc_val)

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
        mi_r_weighted,
        phase_voltages,
        phase_currents,
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
        
        csds.solve_thermal_for_bars(bars_data, HTC=HTC)
        temperature_array = csdf.recreate_temperature_array(bars_data, XSecArray.shape)

        if config["forces"] and any(v != 0 for v in Isc_per_phase.values()):
            csdf.myLog("Computing electromagnetic forces...")
            csdf.compute_forces_for_bars(bars_data, currentsDraw, Isc_per_phase, length)


    # ── Compute per-phase impedance ────────────────────────────────────
    omega = 2 * np.pi * f
    phase_impedance = []   # list of dicts: Zre, Zim, Zmag, Zang, L, R, X
    for U_ph, I_ph, ph_power in zip(phase_voltages, phase_currents, powPh):
        I_rms = abs(I_ph)
        if I_rms > 1e-30:
            Z = U_ph / I_ph          # complex division — Python handles it natively
        else:
            Z = 0 + 0j
        R_ph  = Z.real               # [Ω]
        X_ph  = Z.imag               # [Ω]
        Zmag  = abs(Z)               # [Ω]
        Zang  = np.angle(Z) * 180 / np.pi
        L_ph  = X_ph / omega if omega > 0 else 0   # [H]
        phase_impedance.append(dict(
            Zre=R_ph, Zim=X_ph, Zmag=Zmag, Zang=Zang,
            L=L_ph, Irms=I_rms,
        ))

    # ── Results output ─────────────────────────────────────────────────
    SEP  = "═" * 90
    SEP2 = "─" * 90

    if not simple and not csv:
        print()
        print(SEP)
        print(f"  CDA Results  │  f = {f} Hz  │  L = {length} mm  │  T = {t} °C  │  {config['geometry']}")
        print(SEP)
        print(f"  Materials: {[m.name for m in phases_material]}")
        print(SEP2)

        # Phase power + impedance table
        hdr = (f"  {'Ph':>3}  │  {'I rms (A)':>10}  │  {'Loss (W)':>9}  │"
               f"  {'R (mΩ)':>8}  │  {'X (mΩ)':>8}  │"
               f"  {'|Z| (mΩ)':>9}  │  {'φ (°)':>7}  │  {'L (μH)':>8}")
        print(hdr)
        print(SEP2)
        for ph_idx, (dP, imp) in enumerate(zip(powPh, phase_impedance)):
            print(
                f"  {ph_idx+1:>3}  │  {imp['Irms']:>10.2f}  │  {dP:>9.3f}  │"
                f"  {imp['Zre']*1e3:>8.4f}  │  {imp['Zim']*1e3:>8.4f}  │"
                f"  {imp['Zmag']*1e3:>9.4f}  │  {imp['Zang']:>7.2f}  │  {imp['L']*1e6:>8.4f}"
            )
        print(SEP2)
        print(f"  {'Total':>3}  │  {'':>10}  │  {powerLosses:>9.3f}  W")
        print(SEP)

        if config["bars"]:
            print()
            print("  Individual Conductors")
            print(SEP2)
            show_forces = config["forces"] and any(v != 0 for v in Isc_per_phase.values())
            hdr2 = (f"  {'ID':>4}  │  {'Ph':>3}  │  {'Area (mm²)':>10}  │"
                    f"  {'Perim (mm)':>10}  │  {'Loss (W)':>9}  │"
                    f"  {'I (A)':>8}  │  {'ΔT (K)':>7}")
            if show_forces:
                hdr2 += f"  │  {'Fx (N)':>9}  │  {'Fy (N)':>9}  │  {'|F| (N)':>9}"
            print(hdr2)
            print(SEP2)
            last_ph = None
            for ph_idx, bars in enumerate(phases_conductors):
                for b in bars:
                    bar = bars_data[b - 1]
                    I_bar = abs(bar.current)
                    if bar.phase != last_ph:
                        last_ph = bar.phase
                        isc_label = f"  Isc = {Isc_per_phase.get(bar.phase, 0):.2f} kA" if show_forces else ""
                        print(f"  ── Phase {bar.phase}{isc_label}")
                    line = (f"  {bar.number:>4}  │  {bar.phase:>3}  │  {bar.xsection:>10.2f}  │"
                            f"  {bar.perymiter:>10.2f}  │  {bar.power:>9.3f}  │"
                            f"  {I_bar:>8.2f}  │  {bar.dT:>7.1f}")
                    if show_forces:
                        line += f"  │  {bar.Fx:>+9.3f}  │  {bar.Fy:>+9.3f}  │  {bar.Fmag:>9.3f}"
                    print(line)
            print(SEP2)

    elif not csv:
        print(f"{f} Hz   {powerLosses:.3f} W")
        for ph_idx, (dP, imp) in enumerate(zip(powPh, phase_impedance)):
            print(f"  Ph {ph_idx+1}: {imp['Irms']:.1f} A  {dP:.3f} W  "
                  f"R={imp['Zre']*1e3:.3f} mΩ  X={imp['Zim']*1e3:.3f} mΩ  "
                  f"L={imp['L']*1e6:.4f} μH")
        if config["bars"]:
            show_forces = config["forces"] and any(v != 0 for v in Isc_per_phase.values())
            for b_obj in bars_data:
                line = (f"  [{b_obj.number:>2}] ph{b_obj.phase}  "
                        f"{abs(b_obj.current):>8.2f} A  {b_obj.power:>7.3f} W  "
                        f"{b_obj.perymiter:>7.1f} mm  ΔT={b_obj.dT:>5.1f} K  {b_obj.material.name}")
                if show_forces:
                    line += f"  Fx={b_obj.Fx:+.3f} Fy={b_obj.Fy:+.3f} |F|={b_obj.Fmag:.3f} N"
                print(line)
    else:
        if config["bars"]:
            show_forces = config["forces"] and any(v != 0 for v in Isc_per_phase.values())
            if config["markdown"]:
                hdr_f = " | Fx [N] | Fy [N] | |F| [N]" if show_forces else ""
                print(f"phase | bar | Bar Current [A] | Bar dP [W] | Perimeter [mm] | Center X [mm] | Center Y [mm]{hdr_f}")
                print("---|---|---|---|---|---|---" + ("| --- | --- | ---" if show_forces else ""))
                for bar in bars_data:
                    line = f"{bar.phase}|{bar.number}|{abs(bar.current):.2f}|{bar.power:.3f}|{bar.perymiter:.2f}|{bar.center[0]:.2f}|{bar.center[1]:.2f}"
                    if show_forces:
                        line += f"|{bar.Fx:+.3f}|{bar.Fy:+.3f}|{bar.Fmag:.3f}"
                    print(line)
            else:
                hdr_f = ";Fx [N];Fy [N];|F| [N]" if show_forces else ""
                print(f"phase;bar;Bar Current [A];Bar dP [W];Perimeter [mm];Center X [mm];Center Y [mm]{hdr_f}")
                for bar in bars_data:
                    line = f"{bar.phase};{bar.number};{abs(bar.current):.2f};{bar.power:.3f};{bar.perymiter:.2f};{bar.center[0]:.2f};{bar.center[1]:.2f}"
                    if show_forces:
                        line += f";{bar.Fx:+.3f};{bar.Fy:+.3f};{bar.Fmag:.3f}"
                    print(line)
        else:
            print(f"{f},{powerLosses:.3f}")

    if config["results"]:
        # getting the current density
        resultsCurrentVector *= 1 / (dXmm * dYmm)
        currentsDraw = csdm.recreateresultsArray(
            elementsVector, resultsCurrentVector, XSecArray
        )
        maxCurrent = resultsCurrentVector.max()

        if 1:
            # making the draw of the geometry in initial state.
            base_cmap = plt.get_cmap("inferno", 256)
            colors = base_cmap(np.arange(256))
            colors[0] = [1, 1, 1, 1]  # white background for empty cells
            cmap = ListedColormap(colors)
            norm = PowerNorm(gamma=0.5, vmin=0, vmax=maxCurrent)

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

            # Draw force arrows when forces were computed
            show_forces = config["forces"] and any(v != 0 for v in Isc_per_phase.values())
            if show_forces:
                max_fmag = max((bar.Fmag for bar in bars_data), default=0.0)
                arrow_scale = 40  # pixels (in array-index units) for the longest arrow
                for bar in bars_data:
                    if max_fmag < 1e-12 or bar.Fmag < 1e-12:
                        continue
                    cx = bar.center[0] / dXmm
                    cy = bar.center[1] / dYmm
                    ratio = bar.Fmag / max_fmag
                    adx = (bar.Fx / bar.Fmag) * ratio * arrow_scale
                    ady = (bar.Fy / bar.Fmag) * ratio * arrow_scale
                    ax.annotate(
                        "",
                        xy=(cx + adx, cy + ady),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
                    )
                    ax.text(
                        cx + adx * 0.5,
                        cy + ady * 0.5,
                        f"{bar.Fmag:.1f}N",
                        fontsize=7,
                        color="yellow",
                        ha="center",
                        va="bottom",
                    )

            plt.show()


# Doing the main work here.
if __name__ == "__main__":
    main()
