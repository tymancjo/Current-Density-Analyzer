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
    list_of_phases = [int(n) for n in list_of_phases]
    oryginal_phase_index = {phase: index for index, phase in enumerate(list_of_phases)} 
    # normalizing the phases numbering
    normalized_XsecArr = np.zeros(XSecArray.shape)
    for index,phase in enumerate(list_of_phases):
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

        colors = [all_colors[oryginal_phase_index[n]] for n in list_of_phases]

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
    if len(currents) == number_of_phases - 1:

        Icurrent = [[0, 0] for _ in range(number_of_phases - 1)]
        for i in currents:
            p = oryginal_phase_index[int(i[0])]
            Icurrent[p - 1] = [float(i[1]), float(i[2]) + float(i[3])]
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

    # Reading Material data
    M_list = csdos.read_file_to_list("setup/materials.txt")[1:]
    if M_list:
        MaterialsDB = csdos.get_material_from_list(M_list)
        csdf.myLog(f"Materials: \n {MaterialsDB}")

    phases_material = [0 for _ in range(number_of_phases - 1)]
    if len(materials) == number_of_phases - 1:
        # [phase, mat_number]
        for m in materials:
            index = oryginal_phase_index[int(m[0])] - 1
            index_m = int(m[1])
            if number_of_phases < index or index < 0:
                csdf.myLog("Error! Defined materials for not existing phases!")
                print("Error! Defined materials for not existing phases!")
                sys.exit(1)
            if len(MaterialsDB) < index_m or index_m < 0:
                csdf.myLog("Error! Defined material not id Materials DB file!")
                print("Error! Defined material not id Materials DB file!")
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

        for i, phase in enumerate(phases_conductors):
            for b in phase:
                bars_data[b - 1].phase = i

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
            for i, bars in enumerate(phases_conductors):
                print(f"Phase {i}:")
                phase_curr = 0

                for b in bars:
                    bar = bars_data[b - 1]
                    print(
                        f"\t{bar.current=:.2f} {bar.power=:.2f} {bar.perymiter=:.1f} {bar.center=}"
                    )
                    phase_curr += bar.current
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
                        f"\t{bar.number:>2}\t{csdm.getComplexModule(bar.current):>8.2f}[A]\t{bar.power:>7.2f}[W]\t{bar.perymiter:>7.2f}[mm]"
                    )
                    phase_curr += bar.current
    else:
        if config["bars"]:
            if config["markdown"]:
                print(
                    f"phase | bar | Bar Current [A] | Bar dP[W] | Bar Perymetr[mm] | Bar Center X[mm] | Bar Center Y[mm]"
                )
                print(f"---|---|---|---|---|---|---")
                for bar in bars_data:
                    print(
                        f"{bar.phase}|{bar.number:>2}|{csdm.getComplexModule(bar.current):>8.2f}|{bar.power:>7.2f}|{bar.perymiter:>7.2f}|{bar.center[0]}|{bar.center[1]}"
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
        min_to_draw = maxCurrent/250

        if 1:
            # making the draw of the geometry in initial state.
            base_cmap = plt.get_cmap("jet", 256)
            colors = base_cmap(np.arange(256))
            colors[0] = [1, 1, 1, 1]
            cmap = ListedColormap(colors)
            norm = plt.Normalize(vmin=min_to_draw, vmax=maxCurrent)

            # Adjust the ticks
            # ax = plt.gca()
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 2, width_ratios=[80, 20])
            ax = plt.subplot(gs[0])
            bx = plt.subplot(gs[1])
            bx.axis("off")

            cax = csdf.plot_the_geometry(
                currentsDraw,
                ax,
                cmap,
                dXmm=dXmm,
                dYmm=dYmm,
                norm=norm
            )

            # Add a color bar
            cbar = plt.colorbar(cax, ax=bx)
            cbar.set_label("Current density [A/mm2]", rotation=270, labelpad=20)

            text_line = ""
            for i, dP in enumerate(powPh):
                text_line += f"dP{i}:{dP:.2f}[W] "

            ax.set_title(
                f"I={config['current']}A, f={f}Hz, l={length}mm, Temp={t}degC\n\n\
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
                    text_line = f"[{b:>2}] {csdm.getComplexModule(bar.current):.1f}A\n dP: {bar.power:.1f}W"

                    ax.text(
                        (bar.center[0] / dXmm),
                        bar.center[1] / dYmm,
                        text_line,
                        fontsize=fontsize,
                        color="black",
                    )
            plt.show()


# Doing the main work here.
if __name__ == "__main__":
    main()
