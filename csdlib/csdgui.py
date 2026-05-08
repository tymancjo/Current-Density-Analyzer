"""
This is a tkinter gui lib for CSD library and app
"""

from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox

# Importing local library
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2
from csdlib import csdos

from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm
from csdlib import csdsolve as csds

csdf.verbose = False


class PatternWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Define Pattern")

        self.grid_columnconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.info_label = ctk.CTkLabel(self.main_frame, text="Define the pattern parameters")
        self.info_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.lab_dx = ctk.CTkLabel(self.main_frame, text="Step in X (in cells):")
        self.lab_dx.grid(row=1, column=0, sticky="w", padx=10, pady=2)
        self.idX = ctk.CTkEntry(self.main_frame, width=100)
        self.idX.grid(row=1, column=1, padx=10, pady=2)

        self.lab_dy = ctk.CTkLabel(self.main_frame, text="Step in Y (in cells):")
        self.lab_dy.grid(row=2, column=0, sticky="w", padx=10, pady=2)
        self.idY = ctk.CTkEntry(self.main_frame, width=100)
        self.idY.grid(row=2, column=1, padx=10, pady=2)

        self.lab_n = ctk.CTkLabel(self.main_frame, text="Number of copies:")
        self.lab_n.grid(row=3, column=0, sticky="w", padx=10, pady=2)
        self.iN = ctk.CTkEntry(self.main_frame, width=100)
        self.iN.grid(row=3, column=1, padx=10, pady=2)

        self.submitButton = ctk.CTkButton(self.main_frame, text="Submit", command=self.submit)
        self.submitButton.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    def submit(self):
        try:
            self.dX = int(self.idX.get())
        except ValueError:
            self.dX = 0
        try:
            self.dY = int(self.idY.get())
        except ValueError:
            self.dY = 0
        try:
            self.N = int(self.iN.get())
        except ValueError:
            self.N = 0

        self.destroy()


class GeometryModWindow(ctk.CTkToplevel):
    """
    This is a class that will be used to generate
    a new window for geometry modifications like
    phase replacer or geometry cloning.
    """

    def __init__(self, parent, x_sec_array):
        super().__init__(parent)
        self.title("Geometry Modification")

        self.x_sec_array = x_sec_array

        self.grid_columnconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.info_label = ctk.CTkLabel(self.main_frame, text="Replace Phase")
        self.info_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.source_phase_label = ctk.CTkLabel(self.main_frame, text="Source Phase:")
        self.source_phase_label.grid(row=1, column=0, sticky="w", padx=10, pady=2)
        self.source_phase_var = ctk.StringVar(value="Phase A")
        self.source_phase_menu = ctk.CTkOptionMenu(self.main_frame,
                                                   values=["Phase A", "Phase B", "Phase C"],
                                                   variable=self.source_phase_var)
        self.source_phase_menu.grid(row=1, column=1, padx=10, pady=2)

        self.target_phase_label = ctk.CTkLabel(self.main_frame, text="Target Phase:")
        self.target_phase_label.grid(row=2, column=0, sticky="w", padx=10, pady=2)
        self.target_phase_var = ctk.StringVar(value="Phase B")
        self.target_phase_menu = ctk.CTkOptionMenu(self.main_frame,
                                                   values=["Phase A", "Phase B", "Phase C"],
                                                   variable=self.target_phase_var)
        self.target_phase_menu.grid(row=2, column=1, padx=10, pady=2)

        self.replace_button = ctk.CTkButton(self.main_frame, text="Replace", command=self.replace_phase)
        self.replace_button.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    def replace_phase(self):
        source_phase_str = self.source_phase_var.get()
        target_phase_str = self.target_phase_var.get()

        phase_mapping = {"Phase A": 1, "Phase B": 2, "Phase C": 3}
        source_phase = phase_mapping[source_phase_str]
        target_phase = phase_mapping[target_phase_str]

        if source_phase != target_phase:
            self.x_sec_array[self.x_sec_array == source_phase] = target_phase
            print(f"Phase {source_phase_str} replaced with {target_phase_str}")
            self.destroy()
        else:
            messagebox.showwarning("Warning", "Source and target phases cannot be the same.")


class currentDensityWindowPro(ctk.CTkToplevel):
    """
    This class define the main control window for handling
    the analysis of current density of given geometry.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):
        super().__init__(master)
        self.title("Pro Current Density Solver")

        self.getMaterials()
        print(self.Materials)

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm
        self.lenght = 1000
        self.CuGamma = 391.1  # [W/mK]

        # set grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # create main container frame
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # create tabs
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.tabview.add("Input")
        self.tabview.add("Material")
        self.tabview.add("Thermal")
        
        self.setup_input_tab()
        self.setup_material_tab()
        self.setup_thermal_tab()

        # Console and Buttons Frame
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        self.tx1 = ctk.CTkTextbox(self.bottom_frame, height=120)
        self.tx1.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="nsew")

        self.rButton = ctk.CTkButton(self.bottom_frame, text="Set Parameters", command=self.readSettings)
        self.rButton.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        self.openButton = ctk.CTkButton(self.bottom_frame, text="Calculate!", command=self.powerAnalysis)
        self.openButton.grid(row=1, column=1, padx=10, pady=(5, 10), sticky="ew")

        self.resultsButton = ctk.CTkButton(self.bottom_frame, text="Show Results", command=self.showResults)
        self.resultsButton.grid(row=1, column=2, padx=10, pady=(5, 10), sticky="ew")

        self.isSolved = False
        self.readSettings()

    def setup_input_tab(self):
        tab = self.tabview.tab("Input")
        tab.grid_columnconfigure(0, weight=1)
        
        self.lab_I = ctk.CTkLabel(tab, text="Current RMS [A; deg]")
        self.lab_I.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Irms_txt = ctk.CTkEntry(tab)
        self.Irms_txt.insert(0, "1000;0;1000;120;1000;240")
        self.Irms_txt.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_Freq = ctk.CTkLabel(tab, text="Frequency [Hz]")
        self.lab_Freq.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Freq_txt = ctk.CTkEntry(tab)
        self.Freq_txt.insert(0, "50")
        self.Freq_txt.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_lenght = ctk.CTkLabel(tab, text="Analysis Length [mm]")
        self.lab_lenght.grid(row=4, column=0, padx=10, pady=(10, 2), sticky="w")
        self.lenght_txt = ctk.CTkEntry(tab)
        self.lenght_txt.insert(0, "1000")
        self.lenght_txt.grid(row=5, column=0, padx=10, pady=(0, 10), sticky="ew")
        
    def setup_material_tab(self):
        tab = self.tabview.tab("Material")
        tab.grid_columnconfigure(0, weight=1)

        if self.Materials:
            self.material_buttons_frame = ctk.CTkFrame(tab, fg_color="transparent")
            self.material_buttons_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 10))
            self.material_buttons = []
            for i, M in enumerate(self.Materials):
                self.material_buttons_frame.grid_columnconfigure(i, weight=1)
                button = ctk.CTkButton(
                    self.material_buttons_frame, text=M.name, command=partial(self.setMaterial, M)
                )
                button.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
                self.material_buttons.append(button)

        self.lab_Sigma = ctk.CTkLabel(tab, text="Material conductivity at 20degC [S/m]")
        self.lab_Sigma.grid(row=1, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Sigma_txt = ctk.CTkEntry(tab)
        self.Sigma_txt.insert(0, "56e6")
        self.Sigma_txt.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_temCoRe = ctk.CTkLabel(tab, text="Material Temp coeff of resistance [1/K]")
        self.lab_temCoRe.grid(row=3, column=0, padx=10, pady=(10, 2), sticky="w")
        self.temCoRe_txt = ctk.CTkEntry(tab)
        self.temCoRe_txt.insert(0, "3.9e-3")
        self.temCoRe_txt.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_ro = ctk.CTkLabel(tab, text="material density [kg/m3]")
        self.lab_ro.grid(row=5, column=0, padx=10, pady=(10, 2), sticky="w")
        self.ro_txt = ctk.CTkEntry(tab)
        self.ro_txt.insert(0, "8960")
        self.ro_txt.grid(row=6, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_cp = ctk.CTkLabel(tab, text="material heat capacity [J/kg.K]")
        self.lab_cp.grid(row=7, column=0, padx=10, pady=(10, 2), sticky="w")
        self.cp_txt = ctk.CTkEntry(tab)
        self.cp_txt.insert(0, "385")
        self.cp_txt.grid(row=8, column=0, padx=10, pady=(0, 10), sticky="ew")

    def setup_thermal_tab(self):
        tab = self.tabview.tab("Thermal")
        tab.grid_columnconfigure(0, weight=1)

        self.lab_Temp = ctk.CTkLabel(tab, text="Conductor temperature [degC]")
        self.lab_Temp.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Temp_txt = ctk.CTkEntry(tab)
        self.Temp_txt.insert(0, "140")
        self.Temp_txt.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.lab_HTC = ctk.CTkLabel(tab, text="HTC [W/m2K]")
        self.lab_HTC.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")
        self.HTC_txt = ctk.CTkEntry(tab)
        self.HTC_txt.insert(0, "7")
        self.HTC_txt.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_Gcon = ctk.CTkLabel(tab, text="Thermal Conductivity [W/mK]")
        self.lab_Gcon.grid(row=4, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Gcon_txt = ctk.CTkEntry(tab)
        self.Gcon_txt.insert(0, "0.25")
        self.Gcon_txt.grid(row=5, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_Gmx = ctk.CTkLabel(tab, text="Thermal Conductance Coef. Matrix")
        self.lab_Gmx.grid(row=6, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Gmx_txt = ctk.CTkEntry(tab)
        self.Gmx_txt.insert(0, "0;0;0|0;0;0|0;0;0")
        self.Gmx_txt.grid(row=7, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.lab_Gmx_info = ctk.CTkLabel(tab, text="Fa;Fab;Fac|Fba;Fb;Fbc|Fca;Fcb;Fc")
        self.lab_Gmx_info.grid(row=8, column=0, padx=10, pady=(0, 10), sticky="w")

    def setMaterial(self, M):
        self.Sigma_txt.delete(0, "end")
        self.Sigma_txt.insert(0, M.sigma)

        self.temCoRe_txt.delete(0, "end")
        self.temCoRe_txt.insert(0, M.alpha)

        self.ro_txt.delete(0, "end")
        self.ro_txt.insert(0, M.ro)

        self.cp_txt.delete(0, "end")
        self.cp_txt.insert(0, M.cp)

        self.readSettings()

    def readSettings(self):
        self.I = self.Irms_txt.get().split(";")  # reading I as array
        self.I = [float(I) for I in self.I]

        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())
        self.HTC = float(self.HTC_txt.get())
        self.Gcon = float(self.Gcon_txt.get())
        self.lenght = float(self.lenght_txt.get())
        # the material properties.
        self.sigma20C = float(self.Sigma_txt.get())
        self.temCoRe = float(self.temCoRe_txt.get())
        self.ro = float(self.ro_txt.get())
        self.cp = float(self.cp_txt.get())

        self.Gmx = np.asarray(
            [gx.split(";") for gx in (self.Gmx_txt.get().split("|"))], dtype=float
        )
        self.console("Thermal Conductance Coef. Matrix")
        self.console(self.Gmx)

        # lets workout the  current in phases as is defined
        self.in_Ia = (
            float(self.I[0]) * np.cos(float(self.I[1]) * np.pi / 180)
            + float(self.I[0]) * np.sin(float(self.I[1]) * np.pi / 180) * 1j
        )
        print("in Ia: {}".format(self.in_Ia))

        self.in_Ib = (
            float(self.I[2]) * np.cos(float(self.I[3]) * np.pi / 180)
            + float(self.I[2]) * np.sin(float(self.I[3]) * np.pi / 180) * 1j
        )
        print("in Ib: {}".format(self.in_Ib))

        self.in_Ic = (
            float(self.I[4]) * np.cos(float(self.I[5]) * np.pi / 180)
            + float(self.I[4]) * np.sin(float(self.I[5]) * np.pi / 180) * 1j
        )
        print("in Ic: {}".format(self.in_Ic))

        self.vPhA = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=1, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhB = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=2, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhC = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=3, dXmm=self.dXmm, dYmm=self.dYmm
        )

        # Lets put the all phases together
        self.elementsPhaseA = len(self.vPhA)
        self.elementsPhaseB = len(self.vPhB)
        self.elementsPhaseC = len(self.vPhC)

        if (
            self.elementsPhaseA != 0
            and self.elementsPhaseB != 0
            and self.elementsPhaseC != 0
        ):
            self.elementsVector = np.concatenate(
                (self.vPhA, self.vPhB, self.vPhC), axis=0
            )
        elif self.elementsPhaseA == 0:
            if self.elementsPhaseB == 0:
                self.elementsVector = self.vPhC
            elif self.elementsPhaseC == 0:
                self.elementsVector = self.vPhB
            else:
                self.elementsVector = np.concatenate((self.vPhB, self.vPhC), axis=0)
        else:
            if self.elementsPhaseB == 0 and self.elementsPhaseC == 0:
                self.elementsVector = self.vPhA
            elif self.elementsPhaseC == 0:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhB), axis=0)
            else:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhC), axis=0)

    def console(self, string):
        self.tx1.insert(ctk.END, str(string))
        self.tx1.insert(ctk.END, "\n")
        self.tx1.see(ctk.END)

    def getMaterials(self):
        self.Materials = False
        list = csdos.read_file_to_list("setup/materials.txt")[1:]
        print(list)
        if list:
            self.Materials = csdos.get_material_from_list(list)

    def powerAnalysis(self):
        self.readSettings()

        # moving the calculations to use the function from the module.
        (
            self.resultsCurrentVector,
            powerResults,
            elementsVector,
            powerLossesVector,
        ) = csds.solve_system(
            self.XsecArr,
            self.dXmm,
            self.dYmm,
            self.I,
            self.f,
            self.lenght,
            self.t,
            sigma20C=self.sigma20C,
            temCoRe=self.temCoRe,
        )

        powerLosses, powPhA, powPhB, powPhC = powerResults
        self.solver_length = self.lenght

        self.console(
            "power losses: {:.2f} [W] \n phA: {:.2f}[W]\n phB: {:.2f}[W]\n phC: {:.2f}[W]".format(
                powerLosses, powPhA, powPhB, powPhC
            )
        )

        # self.powerLosses = [powerLosses, powPhA, powPhB, powPhC]
        self.powerLosses = powerResults

        # Doing analysis per bar
        # Checking for the pabrs - separate conductor detecton

        conductors, total, self.phCon = csd.n_getConductors(
            XsecArr=self.XsecArr, vPhA=self.vPhA, vPhB=self.vPhB, vPhC=self.vPhC
        )
        # self.phCon is the list of number of conductors per phase
        print(self.phCon)

        # Going thru the detected bars and preparing the arrays for each of it
        self.bars = []

        for bar in range(1, total + 1):
            temp = csd.n_arrayVectorize(
                inputArray=conductors, phaseNumber=bar, dXmm=self.dXmm, dYmm=self.dYmm
            )
            self.bars.append(temp)

        # Converting resutls to current density
        self.resultsCurrentVector /= self.dXmm * self.dYmm

        # Recreating the solution to form of cross section array
        self.resultsArray = csd.n_recreateresultsArray(
            elementsVector=self.elementsVector,
            resultsVector=self.resultsCurrentVector,
            initialGeometryArray=self.XsecArr,
        )
        self.powerResultsArray = csd.n_recreateresultsArray(
            elementsVector=self.elementsVector,
            resultsVector=powerLossesVector,
            initialGeometryArray=self.XsecArr,
        )

        self.isSolved = True

        # Calculationg the eqivalent single busbar representative object parameters
        # This will be moved to a separate function place in the future

        # Getting the data:
        perymeterA = csd.n_perymiter(self.vPhA, self.XsecArr, self.dXmm, self.dYmm)
        perymeterB = csd.n_perymiter(self.vPhB, self.XsecArr, self.dXmm, self.dYmm)
        perymeterC = csd.n_perymiter(self.vPhC, self.XsecArr, self.dXmm, self.dYmm)

        # temperature coeff of resistance
        alfa = 0.004
        # assuming the thickness of equivalent bar is a=10mm
        a = 10

        b_phA = (perymeterA - 2 * a) / 2
        b_phB = (perymeterB - 2 * a) / 2
        b_phC = (perymeterC - 2 * a) / 2

        # calculating equivalent gamma in 20C - to get the same power losses in DC calculations RI^2
        gamma_phA = (
            (1 + alfa * (self.t - 20))
            * 1
            * float(self.I[0]) ** 2
            / (a * 1e-3 * b_phA * 1e-3 * powPhA)
        )
        gamma_phB = (
            (1 + alfa * (self.t - 20))
            * 1
            * float(self.I[2]) ** 2
            / (a * 1e-3 * b_phB * 1e-3 * powPhB)
        )
        gamma_phC = (
            (1 + alfa * (self.t - 20))
            * 1
            * float(self.I[4]) ** 2
            / (a * 1e-3 * b_phC * 1e-3 * powPhC)
        )

        print("Equivalent bars for DC based thermal analysis: \n")
        print(
            "Eqivalent bar phA is: {}mm x {}mm at gamma: {}".format(a, b_phA, gamma_phA)
        )
        print(
            "Eqivalent bar phB is: {}mm x {}mm at gamma: {}".format(a, b_phB, gamma_phB)
        )
        print(
            "Eqivalent bar phC is: {}mm x {}mm at gamma: {}".format(a, b_phC, gamma_phC)
        )

        print("({},{},1000, gamma={})".format(a, b_phA, gamma_phA))
        print("({},{},1000, gamma={})".format(a, b_phB, gamma_phB))
        print("({},{},1000, gamma={})".format(a, b_phC, gamma_phC))

        # solving the temperatures
        self.calcTempRise()

        # # Display the results:
        # self.showResults()

    def calcTempRise_new(self):
        """
        this procedure solve the thermal equation with given data fromula
        power losses analysis
        """

        # Lets work with barsData for themral model calculations
        if self.isSolved:
            # Doing the power losses sums per each bar
            # Vector to keep all power losses per bar data and perymeter
            # size and temp rise by given HTC

            self.barsData = []

            for i, bar in enumerate(self.bars):
                BarPowerLoss = 0
                BarCurrent = 0

                for element in bar:
                    BarPowerLoss += self.powerResultsArray[
                        int(element[0]), int(element[1])
                    ]

                    BarCurrent += (self.dXmm * self.dYmm) * self.resultsArray[
                        int(element[0]), int(element[1])
                    ]

                # Calculating bar perymiter of the current bar
                perymiter = csd.n_perymiter(bar, self.XsecArr, self.dXmm, self.dYmm)
                center = csd.n_getCenter(bar)
                # print(f"Bar {i} perymiter {perymiter / self.dXmm} \t {perymiter:.2f}mm | Bar center: {center}")

                # Calculating this bar cross section
                XS = len(bar) * self.dXmm * self.dYmm

                # calculationg the bar Ghtc
                p = perymiter * 1e-3
                A = XS * 1e-6
                lng = self.lenght * 1e-3

                Ghtc = p * lng * self.HTC  # thermal conductance to air
                Gt = A * self.CuGamma / lng  # thermal conductance to com
                Q = BarPowerLoss * lng  # Power losses value at lenght

                # Calculating this bar mass
                # for the moment hard coded as copper roCu=8920 [kg/m3]
                roCu = self.ro
                # the heat capacity of copper cpcu=385 [J/kgK]
                cpcu = self.cp

                # its needed for the Icw temp rise calculation
                Vol = A * lng  # [m3]
                Mass = Vol * roCu

                # Calculating the temp rise for 1s
                dT1s = (Q * 1) / (Mass * cpcu)

                # Calculating the temp rise for 3s
                dT3s = (Q * 3) / (Mass * cpcu)

                #  need now to figure out the current phase Number
                if i >= self.phCon[0] + self.phCon[1]:
                    phase = 3
                elif i >= self.phCon[0]:
                    phase = 2
                else:
                    phase = 1

                #  plugin in the data to the list
                self.barsData.append(
                    [center, perymiter, BarCurrent, XS, Q, Ghtc, Gt, phase, dT1s, dT3s]
                )
                # now self.barsData have all the needed info :)

                # barsData structure
                # 0 bar center
                # 1 perymeter
                # 2 bar Current
                # 3 cross section
                # 4 Q power losses value
                # 5 Ghtc to air thermal conductance
                # 6 Gt 1/2lenght thermal conductance
                # 7 phase number

                # 8 New Thermal model DT - this one will calculated later below :)

                # printing data for each bar
                print(
                    "Bar {0:02d} ({5:01d}){1}; Power; {2:06.2f}; [W]; perymeter; {3} [mm]; Current; {4:.1f}; [A]".format(
                        i, center, Q, perymiter, BarCurrent, phase
                    )
                )
                print(
                    "Bar {0:02d} DT(Icu 1s); {1:06.2f}; [K]; DT(Icu 3s); {2:06.2f} [K]".format(
                        i, dT1s, dT3s
                    )
                )

            #  lets figure out the needed size of Gthermal matrix
            #  it will be (bars# +3phases joints)x(the same)
            vectorSize = len(self.barsData) + 3
            thG = np.zeros((vectorSize, vectorSize), dtype=float)

            # TEMP: Hardcoded Gth between matrix

            if self.Gmx.shape != (3, 3):
                GthermalMatrix = np.asarray(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
            else:
                GthermalMatrix = self.Gmx
            # DEBUG
            print("--- Solving for temperatures ---")
            print("The Thermal Cond Coeff Matrix")
            print(GthermalMatrix)

            print("Thermal Conductivity")
            print(self.Gcon)

            print("HTC")
            print(self.HTC)

            print("Results as bars temperatures")

            # now we will loop twice over the bars
            for i, fromBar in enumerate(self.barsData):
                fromPhase = fromBar[7] - 1  # -1 due to the count from 0

                for j, toBar in enumerate(self.barsData):
                    tempG = 0  # just to make sure we dont have something in it

                    if fromBar is toBar:
                        # the main digonal with
                        # GHtc and Gc and sum for all
                        tempG += fromBar[5] + 2 * fromBar[6]
                        #  now we need to loop again all
                        # others to get the sum of G
                        for otherToBar in self.barsData:
                            if otherToBar is not fromBar:
                                #  the distance between to get thermal Conductance
                                distance = (
                                    csd.n_getDistance(fromBar[0], otherToBar[0]) * 1e-3
                                )
                                # the area of the fom Bar as xsection for therm cond
                                thisXs = fromBar[1] * self.lenght * 1e-6

                                otherPhase = otherToBar[7] - 1
                                tempG += (
                                    self.Gcon
                                    * (thisXs / distance)
                                    * GthermalMatrix[fromPhase, otherPhase]
                                )

                    else:
                        #  DEBUG
                        otherPhase = toBar[7] - 1
                        #  the distance between to get thermal Conductance
                        distance = csd.n_getDistance(fromBar[0], toBar[0]) * 1e-3
                        # the area of the fom Bar as xsection for therm cond
                        thisXs = fromBar[1] * self.lenght * 1e-6
                        tempG += (
                            -GthermalMatrix[otherPhase, fromPhase]
                            * self.Gcon
                            * (thisXs / distance)
                        )

                    # putting the calculated vaule in the thG matrix
                    thG[i, j] = tempG

            #  now we need to go for the last 3 rows and columns that
            #  are for the Tx (joints temperatures)
            #  the bar phase will determine which Tx we tackle
            #  Phase = 1 means position -3 in the cols >> col = Phase - 4
            #  so lets go once more thru the bars to fill last columns
            for i, fromBar in enumerate(self.barsData):
                phase = fromBar[7]
                col = phase - 4
                thG[i, col] = -2 * fromBar[6]

            #  and one more to fill the last rows
            for j, fromBar in enumerate(self.barsData):
                phase = fromBar[7]
                row = phase - 4
                thG[row, j] = 2 * fromBar[6]

            # and last thing is the bottom rioght 3x3 area to fill for Tx'es
            # in each phase as sum by bars -2*Gcondution_to_joint
            #  this could be incorporated to the loops above
            #  but is separated for clearer code
            for fromBar in self.barsData:
                phase = fromBar[7]
                col_row = phase - 4
                thG[col_row, col_row] += -2 * fromBar[6]

            #  and one for the Q vector
            thQ = np.zeros((vectorSize), dtype=float)
            for i, fromBar in enumerate(self.barsData):
                thQ[i] = fromBar[4]

            # Solving for the T vector solutions
            thT, _, _, _ = np.linalg.lstsq(thG, thQ, rcond=None)

            # cuts out the Tx joints
            self.Tout = thT[: len(self.barsData)]  # putting result to vector

            # Preparing the output array of the temperatures
            # First we need to rereate vector of temperture for each element
            # in each of bar - as in general solutions vector
            tmpVector = []
            barElemVect = []

            # going thrue each element in each bar
            # creating the long vetor of temp risies
            # and properly ordered elements vector
            # that render where in oryginal xsec array was the element

            for i, bar in enumerate(self.bars):
                for element in bar:
                    tmpVector.append(self.Tout[i])
                    barElemVect.append(element)

            # Now we prepare the array to display
            self.tempriseResultsArray = csd.n_recreateresultsArray(
                elementsVector=barElemVect,
                resultsVector=tmpVector,
                initialGeometryArray=self.XsecArr,
            )

            for i, temp in enumerate(self.Tout):
                self.barsData[i].append(temp)
                print("Bar {}: {:.2f}[K]".format(i, temp))

            print("Phase A joint: {:.2f}[K]".format(thT[-3]))
            print("Phase B joint: {:.2f}[K]".format(thT[-2]))
            print("Phase C joint: {:.2f}[K]".format(thT[-1]))

            # and now remembering all thermal results
            self.Tout = thT

            # Added 05.11.2019 - Adiabatic temperature rise (Icw) analysis for each Bar
            # Listng results for the Icw DT calculations
            avT = 0
            print("****** 1s Icw Adiabatic Temp Rise *******")
            for i, barDT in enumerate(self.barsData):
                print(
                    "Bar {}: {:.2f}[K]  ({:.2f} degC@35)".format(
                        i, barDT[8], barDT[8] + 35
                    )
                )
                avT += barDT[8] + 35

            avT = avT / len(self.barsData)
            print("Average 1s: {:.2f} degC@35".format(avT))
            avT = 0
            print("****** 3s Icw Adiabatic Temp Rise *******")
            for i, barDT in enumerate(self.barsData):
                print(
                    "Bar {}: {:.2f}[K]  ({:.2f} degC@35)".format(
                        i, barDT[9], barDT[9] + 35
                    )
                )
                avT += barDT[9] + 35

            avT = avT / len(self.barsData)
            print("Average 3s: {:.2f} degC@35".format(avT))

            print("******* END Icw Adiabatic Temp Rise *****")

            print("####### FOR PCG #######")
            for i, bar in enumerate(self.barsData):
                print(f"{i+1:02d} dP: \t {bar[4]:7.2f} W")

            # Display the results:
            self.showResults()

    def calcTempRise(self):
        """
        this procedure solve the thermal equation with given data fromula
        power losses analysis
        """

        # Lets work with barsData for themral model calculations
        if self.isSolved:
            # Doing the power losses sums per each bar
            # Vector to keep all power losses per bar data and perymeter
            # size and temp rise by given HTC

            self.barsData = []

            for i, bar in enumerate(self.bars):
                BarPowerLoss = 0
                BarCurrent = 0

                # print(len(bar))

                for element in bar:
                    BarPowerLoss += self.powerResultsArray[
                        int(element[0]), int(element[1])
                    ]

                    BarCurrent += (self.dXmm * self.dYmm) * self.resultsArray[
                        int(element[0]), int(element[1])
                    ]

                # Calculating bar perymiter of the current bar
                perymiter = csd.n_perymiter(bar, self.XsecArr, self.dXmm, self.dYmm)
                center = csd.n_getCenter(bar)
                # Calculating this bar cross section
                XS = len(bar) * self.dXmm * self.dYmm

                # calculationg the bar Ghtc
                p = perymiter * 1e-3
                A = XS * 1e-6
                lng = self.lenght * 1e-3

                Ghtc = p * lng * self.HTC  # thermal conductance to air
                Gt = A * self.CuGamma / lng  # thermal conductance to com
                Q = BarPowerLoss * self.lenght / self.solver_length   # Power losses value at lenght that was analysed brought to the temp analysis length 

                # Calculating this bar mass
                # for the moment hard coded as copper roCu=8920 [kg/m3]
                roCu = self.ro
                # the heat capacity of copper cpcu=385 [J/kgK]
                cpcu = self.cp

                # its needed for the Icw temp rise calculation
                Vol = A * lng  # [m3]
                Mass = Vol * roCu

                # Calculating the temp rise for 1s
                dT1s = (Q * 1) / (Mass * cpcu)

                # Calculating the temp rise for 3s
                dT3s = (Q * 3) / (Mass * cpcu)

                #  need now to figure out the current phase Number
                if i >= self.phCon[0] + self.phCon[1]:
                    phase = 3
                elif i >= self.phCon[0]:
                    phase = 2
                else:
                    phase = 1

                #  plugin in the data to the list
                self.barsData.append(
                    [center, perymiter, BarCurrent, XS, Q, Ghtc, Gt, phase, dT1s, dT3s]
                )
                # now self.barsData have all the needed info :)

                # barsData structure
                # 0 bar center
                # 1 perymeter
                # 2 bar Current
                # 3 cross section
                # 4 Q power losses value
                # 5 Ghtc to air thermal conductance
                # 6 Gt 1/2lenght thermal conductance
                # 7 phase number

                # 8 New Thermal model DT - this one will calculated later below :)

                # printing data for each bar
                print(
                    "Bar {0:02d} ({5:01d}){1}; Power; {2:06.2f}; [W]; perymeter; {3} [mm]; Current; {4:.1f}; [A]".format(
                        i, center, Q, perymiter, BarCurrent, phase
                    )
                )
                print(
                    "Bar {0:02d} DT(Icu 1s); {1:06.2f}; [K]; DT(Icu 3s); {2:06.2f} [K]".format(
                        i, dT1s, dT3s
                    )
                )

            # print('** Bars Data **')
            # print(self.barsData)
            # print('** Bars Data **')

            #  lets figure out the needed size of Gthermal matrix
            #  it will be (bars# +3phases joints)x(the same)
            vectorSize = len(self.barsData) + 3
            thG = np.zeros((vectorSize, vectorSize), dtype=float)

            # TEMP: Hardcoded Gth between matrix

            if self.Gmx.shape != (3, 3):
                GthermalMatrix = np.asarray(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
            else:
                GthermalMatrix = self.Gmx
            # DEBUG
            print("--- Solving for temperatures ---")
            print("The Thermal Cond Coeff Matrix")
            print(GthermalMatrix)

            print("Thermal Conductivity")
            print(self.Gcon)

            print("HTC")
            print(self.HTC)

            print("Results as bars temperatures")

            # now we will loop twice over the bars
            for i, fromBar in enumerate(self.barsData):
                fromPhase = fromBar[7] - 1  # -1 due to the count from 0

                for j, toBar in enumerate(self.barsData):
                    tempG = 0  # just to make sure we dont have something in it

                    if fromBar is toBar:
                        # the main digonal with
                        # GHtc and Gc and sum for all

                        # DEBUG
                        # print('({},{}) it is me!'.format(i,j))
                        tempG += fromBar[5] + 2 * fromBar[6]
                        #  now we nwwd to loop again all
                        # others to get the sum of G
                        for otherToBar in self.barsData:
                            if otherToBar is not fromBar:
                                #  the distance between to get thermal Conductance
                                distance = (
                                    csd.n_getDistance(fromBar[0], otherToBar[0]) * 1e-3
                                )
                                # the area of the fom Bar as xsection for therm cond
                                thisXs = fromBar[1] * self.lenght * 1e-6

                                otherPhase = otherToBar[7] - 1
                                tempG += (
                                    self.Gcon
                                    * (thisXs / distance)
                                    * GthermalMatrix[fromPhase, otherPhase]
                                )

                    else:
                        #  DEBUG
                        # print('({},{}) someone else'.format(i,j))
                        otherPhase = toBar[7] - 1
                        #  the distance between to get thermal Conductance
                        distance = csd.n_getDistance(fromBar[0], toBar[0]) * 1e-3
                        # the area of the fom Bar as xsection for therm cond
                        thisXs = fromBar[1] * self.lenght * 1e-6
                        tempG += (
                            -GthermalMatrix[otherPhase, fromPhase]
                            * self.Gcon
                            * (thisXs / distance)
                        )

                    # putting the calculated vaule in the thG matrix
                    thG[i, j] = tempG

            #  now we need to go for the last 3 rows and columns that
            #  are for the Tx (joints temperatures)
            #  the bar phase will determine which Tx we tackle
            #  Phase = 1 means position -3 in the cols >> col = Phase - 4
            #  so lets go once more thru the bars to fill last columns
            for i, fromBar in enumerate(self.barsData):
                phase = fromBar[7]
                col = phase - 4
                thG[i, col] = -2 * fromBar[6]

            #  and one more to fill the last rows
            for j, fromBar in enumerate(self.barsData):
                phase = fromBar[7]
                row = phase - 4
                thG[row, j] = 2 * fromBar[6]

            # and last thing is the bottom rioght 3x3 area to fill for Tx'es
            # in each phase as sum by bars -2*Gcondution_to_joint
            #  this could be incorporated to the loops above
            #  but is separated for clearer code
            for fromBar in self.barsData:
                phase = fromBar[7]
                col_row = phase - 4
                thG[col_row, col_row] += -2 * fromBar[6]

            #  and one for the Q vector
            thQ = np.zeros((vectorSize), dtype=float)
            for i, fromBar in enumerate(self.barsData):
                thQ[i] = fromBar[4]

            # Solving for thT vector solutions
            thT, _, _, _ = np.linalg.lstsq(thG, thQ, rcond=None)

            #  DEBUG
            # print('The G array')
            # print(thG)
            # print('The Q vector')
            # print(thQ)
            # print('The T vector')
            # print(thT)

            # cuts out the Tx joints
            self.Tout = thT[: len(self.barsData)]  # putting result to vector

            # Preparing the output array of the temperatures
            # First we need to rereate vector of temperture for each element
            # in each of bar - as in general solutions vector
            tmpVector = []
            barElemVect = []

            # going thrue each element in each bar
            # creating the long vetor of temp risies
            # and properly ordered elements vector
            # that render where in oryginal xsec array was the element

            for i, bar in enumerate(self.bars):
                for element in bar:
                    tmpVector.append(self.Tout[i])
                    barElemVect.append(element)

            # Now we prepare the array to display
            self.tempriseResultsArray = csd.n_recreateresultsArray(
                elementsVector=barElemVect,
                resultsVector=tmpVector,
                initialGeometryArray=self.XsecArr,
            )

            for i, temp in enumerate(self.Tout):
                self.barsData[i].append(temp)
                print("Bar {}: {:.2f}[K]".format(i, temp))

            print("Phase A joint: {:.2f}[K]".format(thT[-3]))
            print("Phase B joint: {:.2f}[K]".format(thT[-2]))
            print("Phase C joint: {:.2f}[K]".format(thT[-1]))

            # and now remembering all thermal results
            self.Tout = thT

            # Added 05.11.2019 - Adiabatic temperature rise (Icw) analysis for each Bar
            # Listng results for the Icw DT calculations
            avT = 0
            print("****** 1s Icw Adiabatic Temp Rise *******")
            for i, barDT in enumerate(self.barsData):
                print(
                    "Bar {}: {:.2f}[K]  ({:.2f} degC@35)".format(
                        i, barDT[8], barDT[8] + 35
                    )
                )
                avT += barDT[8] + 35

            avT = avT / len(self.barsData)
            print("Average 1s: {:.2f} degC@35".format(avT))
            avT = 0
            print("****** 3s Icw Adiabatic Temp Rise *******")
            for i, barDT in enumerate(self.barsData):
                print(
                    "Bar {}: {:.2f}[K]  ({:.2f} degC@35)".format(
                        i, barDT[9], barDT[9] + 35
                    )
                )
                avT += barDT[9] + 35

            avT = avT / len(self.barsData)
            print("Average 3s: {:.2f} degC@35".format(avT))

            print("******* END Icw Adiabatic Temp Rise *****")

            print("####### FOR PCG #######")
            for i, bar in enumerate(self.barsData):
                print(f"{i+1:02d} dP: \t {bar[4]:7.2f} W")

            # Display the results:
            self.showResults()

    def showResults(self):
        title_font = {"size": "11", "color": "black", "weight": "normal"}
        axis_font = {"size": "10"}
        show_bars_no = False

        if np.sum(self.resultsArray) != 0:
            # Cecking the area in array that is used by geometry to limit the display
            min_row = int(np.min(self.elementsVector[:, 0]))
            max_row = int(np.max(self.elementsVector[:, 0]) + 1)

            min_col = int(np.min(self.elementsVector[:, 1]))
            max_col = int(np.max(self.elementsVector[:, 1]) + 1)

            # Cutting down results array to the area with geometry
            tempriseArrayDisplay = self.tempriseResultsArray[
                min_row:max_row, min_col:max_col
            ]
            resultsArrayDisplay = self.resultsArray[min_row:max_row, min_col:max_col]

            # Checking out what are the dimensions od the ploted area
            # to make propper scaling

            plotWidth = (resultsArrayDisplay.shape[1]) * self.dXmm
            plotHeight = (resultsArrayDisplay.shape[0]) * self.dYmm

            fig = plt.figure("Power Results Window")
            ax = fig.add_subplot(1, 1, 1)

            my_cmap = matplotlib.cm.get_cmap("jet")
            my_cmap.set_under("w")

            im = ax.imshow(
                resultsArrayDisplay,
                cmap=my_cmap,
                interpolation="none",
                vmin=0.8 * np.min(self.resultsCurrentVector),
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

            # Putting the detected bars numvers on plot to reffer the console data
            # And doing calculation for each bar

            if show_bars_no:
                for i, bar in enumerate(self.bars):
                    x, y = csd.n_getCenter(bar)
                    x -= min_col * self.dXmm
                    y -= min_row * self.dYmm

                    ax.text(x, y, "[{}]".format(i), horizontalalignment="center")
                    # self.console('bar {0:02d}: {1:.01f}[K]'.format(i, self.barsData[i][6]))

            # *** end of the per bar analysis ***

            ax.set_title(
                str(self.f)
                + "[Hz] / "
                + str(self.I)
                + "[A] / "
                + str(self.t)
                + "[$^o$C] /"
                + str(self.lenght)
                + "[mm]\n Power Losses {0[0]:.2f}[W] \n phA: {0[1]:.2f} phB: {0[2]:.2f} phC: {0[3]:.2f}".format(
                    self.powerLosses
                ),
                **title_font,
            )

            plt.xlabel("size [mm]", **axis_font)
            plt.ylabel("size [mm]", **axis_font)

            fig.autofmt_xdate(bottom=0.2, rotation=45, ha="right")

            plt.tight_layout()

            self.showTemperatureResults()
            plt.show()

    def showTemperatureResults(self):
        title_font = {"size": "9", "color": "black", "weight": "normal"}
        axis_font = {"size": "9"}

        if np.sum(self.resultsArray) != 0:
            # Cecking the area in array that is used by geometry to limit the display
            min_row = int(np.min(self.elementsVector[:, 0]))
            max_row = int(np.max(self.elementsVector[:, 0]) + 1)

            min_col = int(np.min(self.elementsVector[:, 1]))
            max_col = int(np.max(self.elementsVector[:, 1]) + 1)

            # Cutting down results array to the area with geometry
            resultsArrayDisplay = self.tempriseResultsArray[
                min_row:max_row, min_col:max_col
            ]

            # Checking out what are the dimensions od the ploted area
            # to make propper scaling

            plotWidth = (resultsArrayDisplay.shape[1]) * self.dXmm
            plotHeight = (resultsArrayDisplay.shape[0]) * self.dYmm

            fig = plt.figure("Temperature Results Window")
            ax = fig.add_subplot(1, 1, 1)

            my_cmap = matplotlib.cm.get_cmap("jet")
            my_cmap.set_under("w")

            im = ax.imshow(
                resultsArrayDisplay,
                cmap=my_cmap,
                interpolation="none",
                vmin=0.8 * np.min(self.Tout),
                extent=[0, plotWidth, plotHeight, 0],
            )

            fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                label="Temperature Rise [K]",
                alpha=0.5,
                fraction=0.046,
            )
            plt.axis("scaled")

            # Putting the detected bars numvers on plot to reffer the console data
            # And doing calculation for each bar

            for i, bar in enumerate(self.bars):
                x, y = csd.n_getCenter(bar)
                x -= min_col * self.dXmm
                y -= min_row * self.dYmm

                # DT = "[{}]\n1s: {:.2f}\n 3s: {:.2f}".format(
                #     i, self.barsData[i][8] + 35, self.barsData[i][9] + 35
                # )
                DT = f"[{self.barsData[i][2]:.0f} A]"

                ax.text(
                    x,
                    y,
                    DT,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8,
                )

            # *** end of the per bar analysis ***

            ax.set_title(
                str(self.f)
                + "[Hz] /"
                + str(self.t)
                + "[$^o$C] /"
                + str(self.lenght)
                + "[mm] \n" "Ia:{:.1f}A {:.0f}$^o$ ".format(
                    float(self.I[0]), np.floor(float(self.I[1]))
                )
                + "Ib:{:.1f}A {:.0f}$^o$ ".format(
                    float(self.I[2]), np.floor(float(self.I[3]))
                )
                + "Ic:{:.1f}A {:.0f}$^o$ \n".format(
                    float(self.I[4]), np.floor(float(self.I[5]))
                )
                + "HTC: {}[W/m$^2$K] / ThermConv: {}[W/mK]".format(self.HTC, self.Gcon)
                + "\n Joints Temp Rises: Fa:{:.2f}K Fb;{:.2f}K Fc:{:.2f}K".format(
                    self.Tout[-3], self.Tout[-2], self.Tout[-1]
                ),
                **title_font,
            )

            plt.xlabel("size [mm]", **axis_font)
            plt.ylabel("size [mm]", **axis_font)

            fig.autofmt_xdate(bottom=0.2, rotation=45, ha="right")

            plt.tight_layout()
            # plt.show()


class zWindow(ctk.CTkToplevel):
    """
    This class define the main control window for handling
    the analysis of equivalent phase impedance of given geometry.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):
        super().__init__(master)
        self.title("Impedance Calculator")

        self.getMaterials()
        print(self.Materials)

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        # set grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # create main container frame
        self.container_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.container_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.container_frame.grid_columnconfigure(0, weight=1)
        self.container_frame.grid_columnconfigure(1, weight=2)
        
        # Input Frame
        self.input_frame = ctk.CTkFrame(self.container_frame)
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.lab_Freq = ctk.CTkLabel(self.input_frame, text="Frequency [Hz]")
        self.lab_Freq.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Freq_txt = ctk.CTkEntry(self.input_frame)
        self.Freq_txt.insert(0, "50")
        self.Freq_txt.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_Temp = ctk.CTkLabel(self.input_frame, text="Conductor temperature [degC]")
        self.lab_Temp.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Temp_txt = ctk.CTkEntry(self.input_frame)
        self.Temp_txt.insert(0, "140")
        self.Temp_txt.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Summary Frame
        self.summary_frame = ctk.CTkFrame(self.container_frame)
        self.summary_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.summary_frame.grid_columnconfigure(0, weight=1)

        self.desc_f = ctk.CTkLabel(self.summary_frame, text="Frequency: 50.00 [Hz]")
        self.desc_f.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.desc_t = ctk.CTkLabel(self.summary_frame, text="Temperature: 140.00 [degC]")
        self.desc_t.grid(row=1, column=0, padx=10, pady=(2, 10), sticky="w")

        # Materials Frame
        self.materials_frame = ctk.CTkFrame(self.container_frame)
        self.materials_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=5)
        self.materials_frame.grid_columnconfigure(0, weight=1)
        
        self.materials_label = ctk.CTkLabel(self.materials_frame, text="Materials")
        self.materials_label.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")

        if self.Materials:
            self.material_buttons_frame = ctk.CTkFrame(self.materials_frame, fg_color="transparent")
            self.material_buttons_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
            self.material_buttons = []
            for i, M in enumerate(self.Materials):
                self.material_buttons_frame.grid_columnconfigure(i, weight=1)
                button = ctk.CTkButton(
                    self.material_buttons_frame, text=M.name, command=partial(self.setMaterial, M)
                )
                button.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
                self.material_buttons.append(button)

            self.M = self.Materials[0]

        # Console and Buttons Frame
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        self.tx1 = ctk.CTkTextbox(self.bottom_frame, height=120)
        self.tx1.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="nsew")

        self.rButton = ctk.CTkButton(
            self.bottom_frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        self.openButton = ctk.CTkButton(
            self.bottom_frame, text="Calculate!", command=self.powerAnalysis
        )
        self.openButton.grid(row=1, column=1, padx=10, pady=(5, 10), sticky="ew")
        
        # Set initial values
        self.readSettings()

    def readSettings(self):
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f.configure(text="Frequency: {:.2f} [Hz]".format(self.f))
        self.desc_t.configure(text="Temperature: {:.2f} [degC]".format(self.t))

        self.vPhA = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=1, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhB = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=2, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhC = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=3, dXmm=self.dXmm, dYmm=self.dYmm
        )

        self.elementsPhaseA = len(self.vPhA)
        self.elementsPhaseB = len(self.vPhB)
        self.elementsPhaseC = len(self.vPhC)

        if (
            self.elementsPhaseA != 0
            and self.elementsPhaseB != 0
            and self.elementsPhaseC != 0
        ):
            self.elementsVector = np.concatenate(
                (self.vPhA, self.vPhB, self.vPhC), axis=0
            )
        elif self.elementsPhaseA == 0:
            if self.elementsPhaseB == 0:
                self.elementsVector = self.vPhC
            elif self.elementsPhaseC == 0:
                self.elementsVector = self.vPhB
            else:
                self.elementsVector = np.concatenate((self.vPhB, self.vPhC), axis=0)
        else:
            if self.elementsPhaseB == 0 and self.elementsPhaseC == 0:
                self.elementsVector = self.vPhA
            elif self.elementsPhaseC == 0:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhB), axis=0)
            else:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhC), axis=0)

    def getMaterials(self):
        self.Materials = False
        list = csdos.read_file_to_list("setup/materials.txt")[1:]
        print(list)
        if list:
            self.Materials = csdos.get_material_from_list(list)

    def setMaterial(self, M):
        self.M = M
        self.console(f"Set to use: {M.name} as material")

    def console(self, string):
        self.tx1.insert(ctk.END, str(string))
        self.tx1.insert(ctk.END, "\n")
        self.tx1.see(ctk.END)

    def powerAnalysis(self):
        self.readSettings()

        # Let's put here some voltage vector
        # initial voltage values
        Ua = complex(1, 0)
        Ub = complex(-0.5, np.sqrt(3) / 2)
        Uc = complex(-0.5, -np.sqrt(3) / 2)

        admitanceMatrix = np.linalg.inv(
            csd.n_getImpedanceArray(
                csd.n_getDistancesArray(self.vPhA),
                freq=self.f,
                dXmm=self.dXmm,
                dYmm=self.dYmm,
                temperature=self.t,
                # lenght=self.lenght,
                sigma20C=self.M.sigma,
                temCoRe=self.M.alpha,
            )
        )

        voltageVector = np.ones(len(self.vPhA)) * Ua

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        Ia = np.sum(currentVector)

        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Za = Ua / Ia
        La = Za.imag / (2 * np.pi * self.f)

        # round 2 - phase B - other phases shunted

        admitanceMatrix = np.linalg.inv(
            csd.n_getImpedanceArray(
                csd.n_getDistancesArray(self.vPhB),
                freq=self.f,
                dXmm=self.dXmm,
                dYmm=self.dYmm,
                temperature=self.t,
                # lenght=self.lenght,
                sigma20C=self.M.sigma,
                temCoRe=self.M.alpha,
            )
        )

        voltageVector = np.ones(len(self.vPhB)) * Ub

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        Ib = np.sum(currentVector)

        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Zb = Ub / Ib
        Lb = Zb.imag / (2 * np.pi * self.f)

        # round 3 - phase C - other phases shunted
        admitanceMatrix = np.linalg.inv(
            csd.n_getImpedanceArray(
                csd.n_getDistancesArray(self.vPhC),
                freq=self.f,
                dXmm=self.dXmm,
                dYmm=self.dYmm,
                temperature=self.t,
                # lenght=self.lenght,
                sigma20C=self.M.sigma,
                temCoRe=self.M.alpha,
            )
        )

        voltageVector = np.ones(len(self.vPhC)) * Uc

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        Ic = np.sum(currentVector)

        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Zc = Uc / Ic
        Lc = Zc.imag / (2 * np.pi * self.f)

        print("Impedance calulations results: \n")
        print("Za: {:.2f}  [uOhm]  La = {:.3f} [uH]".format(Za * 1e6, La * 1e6))
        print("Zb: {:.2f}  [uOhm]  Lb = {:.3f} [uH]".format(Zb * 1e6, Lb * 1e6))
        print("Zc: {:.2f}  [uOhm]  Lc = {:.3f} [uH]".format(Zc * 1e6, Lc * 1e6))
        print("########################################################\n \n")

        # printing to GUI console window
        self.console("Impedance calulations results:")
        self.console("Za: {:.2f} [uOhm]".format(Za * 1e6))
        self.console("Zb: {:.2f} [uOhm]".format(Zb * 1e6))
        self.console("Zc: {:.2f} [uOhm]".format(Zc * 1e6))
        self.console("")
        self.console("La: {:.3f} [uH]".format(La * 1e6))
        self.console("Lb: {:.3f} [uH]".format(Lb * 1e6))
        self.console("Lc: {:.3f} [uH]".format(Lc * 1e6))


class zWindow3f(ctk.CTkToplevel):
    """
    This class define the main control window for handling
    the analysis of equivalent phase impedance of given geometry in case of 3f shunted scenario.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):
        super().__init__(master)
        self.title("3-Phase Impedance Calculator")

        self.getMaterials()
        print(self.Materials)

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        # set grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # create main container frame
        self.container_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.container_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.container_frame.grid_columnconfigure(0, weight=1)
        self.container_frame.grid_columnconfigure(1, weight=2)
        
        # Input Frame
        self.input_frame = ctk.CTkFrame(self.container_frame)
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.lab_Freq = ctk.CTkLabel(self.input_frame, text="Frequency [Hz]")
        self.lab_Freq.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Freq_txt = ctk.CTkEntry(self.input_frame)
        self.Freq_txt.insert(0, "50")
        self.Freq_txt.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_Temp = ctk.CTkLabel(self.input_frame, text="Conductor temperature [degC]")
        self.lab_Temp.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Temp_txt = ctk.CTkEntry(self.input_frame)
        self.Temp_txt.insert(0, "140")
        self.Temp_txt.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Summary Frame
        self.summary_frame = ctk.CTkFrame(self.container_frame)
        self.summary_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.summary_frame.grid_columnconfigure(0, weight=1)

        self.desc_f = ctk.CTkLabel(self.summary_frame, text="Frequency: 50.00 [Hz]")
        self.desc_f.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.desc_t = ctk.CTkLabel(self.summary_frame, text="Temperature: 140.00 [degC]")
        self.desc_t.grid(row=1, column=0, padx=10, pady=(2, 10), sticky="w")

        # Materials Frame
        self.materials_frame = ctk.CTkFrame(self.container_frame)
        self.materials_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=5)
        self.materials_frame.grid_columnconfigure(0, weight=1)
        
        self.materials_label = ctk.CTkLabel(self.materials_frame, text="Materials")
        self.materials_label.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")

        if self.Materials:
            self.material_buttons_frame = ctk.CTkFrame(self.materials_frame, fg_color="transparent")
            self.material_buttons_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
            self.material_buttons = []
            for i, M in enumerate(self.Materials):
                self.material_buttons_frame.grid_columnconfigure(i, weight=1)
                button = ctk.CTkButton(
                    self.material_buttons_frame, text=M.name, command=partial(self.setMaterial, M)
                )
                button.grid(row=0, column=i, padx=2, pady=2, sticky="ew")
                self.material_buttons.append(button)

            self.M = self.Materials[0]

        # Console and Buttons Frame
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        self.tx1 = ctk.CTkTextbox(self.bottom_frame, height=120)
        self.tx1.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="nsew")

        self.rButton = ctk.CTkButton(
            self.bottom_frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        self.openButton = ctk.CTkButton(
            self.bottom_frame, text="Calculate!", command=self.powerAnalysis
        )
        self.openButton.grid(row=1, column=1, padx=10, pady=(5, 10), sticky="ew")
        
        # Set initial values
        self.readSettings()

    def readSettings(self):
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f.configure(text="Frequency: {:.2f} [Hz]".format(self.f))
        self.desc_t.configure(text="Temperature: {:.2f} [degC]".format(self.t))

        self.vPhA = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=1, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhB = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=2, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhC = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=3, dXmm=self.dXmm, dYmm=self.dYmm
        )

        self.elementsPhaseA = len(self.vPhA)
        self.elementsPhaseB = len(self.vPhB)
        self.elementsPhaseC = len(self.vPhC)

        if (
            self.elementsPhaseA != 0
            and self.elementsPhaseB != 0
            and self.elementsPhaseC != 0
        ):
            self.elementsVector = np.concatenate(
                (self.vPhA, self.vPhB, self.vPhC), axis=0
            )
        elif self.elementsPhaseA == 0:
            if self.elementsPhaseB == 0:
                self.elementsVector = self.vPhC
            elif self.elementsPhaseC == 0:
                self.elementsVector = self.vPhB
            else:
                self.elementsVector = np.concatenate((self.vPhB, self.vPhC), axis=0)
        else:
            if self.elementsPhaseB == 0 and self.elementsPhaseC == 0:
                self.elementsVector = self.vPhA
            elif self.elementsPhaseC == 0:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhB), axis=0)
            else:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhC), axis=0)

    def console(self, string):
        self.tx1.insert(ctk.END, str(string))
        self.tx1.insert(ctk.END, "\n")
        self.tx1.see(ctk.END)

    def getMaterials(self):
        self.Materials = False
        list = csdos.read_file_to_list("setup/materials.txt")[1:]
        print(list)
        if list:
            self.Materials = csdos.get_material_from_list(list)

    def setMaterial(self, M):
        self.M = M
        self.console(f"Set to use: {M.name} as material")

    def powerAnalysis(self):
        self.readSettings()

        admitanceMatrix = np.linalg.inv(
            csd.n_getImpedanceArray(
                csd.n_getDistancesArray(self.elementsVector),
                freq=self.f,
                dXmm=self.dXmm,
                dYmm=self.dYmm,
                temperature=self.t,
                # lenght=self.lenght,
                sigma20C=self.M.sigma,
                temCoRe=self.M.alpha,
            )
        )

        # Let's put here some voltage vector
        Ua = complex(1, 0)
        Ub = complex(-0.5, np.sqrt(3) / 2)
        Uc = complex(-0.5, -np.sqrt(3) / 2)

        vA = np.ones(self.elementsPhaseA) * Ua
        vB = np.ones(self.elementsPhaseB) * Ub
        vC = np.ones(self.elementsPhaseC) * Uc

        voltageVector = np.concatenate((vA, vB, vC), axis=0)

        # Initial solve
        # Main equation solve
        currentVector = np.matmul(admitanceMatrix, voltageVector)

        # And now we need to get solution for each phase to normalize it
        currentPhA = currentVector[0 : self.elementsPhaseA]
        currentPhB = currentVector[
            self.elementsPhaseA : self.elementsPhaseA + self.elementsPhaseB
        ]
        currentPhC = currentVector[self.elementsPhaseA + self.elementsPhaseB :]

        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)
        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Za = Ua / Ia
        La = Za.imag / (2 * np.pi * self.f)

        Zb = Ub / Ib
        Lb = Zb.imag / (2 * np.pi * self.f)

        Zc = Uc / Ic
        Lc = Zc.imag / (2 * np.pi * self.f)

        print("Current results:")
        print(f"Ia: {Ia}")
        print(f"Ib: {Ib}")
        print(f"Ic: {Ic}")
        print("")
        print("Impedance calulations results: \n")
        print(
            f"Za: {Za*1e6:.2f}  [uOhm]  |Za|: {abs(Za*1e6):.2f} La = {La*1e6:.3f} [uH]"
        )
        print(
            f"Zb: {Zb*1e6:.2f}  [uOhm]  |Zb|: {abs(Zb*1e6):.2f} La = {Lb*1e6:.3f} [uH]"
        )
        print(
            f"Zc: {Zc*1e6:.2f}  [uOhm]  |Zc|: {abs(Zc*1e6):.2f} La = {Lc*1e6:.3f} [uH]"
        )
        print("########################################################\n \n")

        # printing to GUI console window
        self.console("Impedance calulations results:")
        self.console(f"Za: {Za*1e6:.2f}  [uOhm]  |Za|: {abs(Za*1e6):.2f}")
        self.console(f"Zb: {Zb*1e6:.2f}  [uOhm]  |Zb|: {abs(Zb*1e6):.2f}")
        self.console(f"Zc: {Zc*1e6:.2f}  [uOhm]  |Zc|: {abs(Zc*1e6):.2f}")

        self.console("")
        self.console("La: {:.3f} [uH]".format(La * 1e6))
        self.console("Lb: {:.3f} [uH]".format(Lb * 1e6))
        self.console("Lc: {:.3f} [uH]".format(Lc * 1e6))


class forceWindow(ctk.CTkToplevel):
    """
    This class define the main control window for the
    electrodynamic forces analysis.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):
        super().__init__(master)
        self.title("Forces calculator")

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        # set grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # create main container frame
        self.container_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.container_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.container_frame.grid_columnconfigure((0, 1), weight=1)

        # Input Frame
        self.input_frame = ctk.CTkFrame(self.container_frame)
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.lab_l = ctk.CTkLabel(self.input_frame, text="Analysis length [mm]")
        self.lab_l.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.lenght = ctk.CTkEntry(self.input_frame)
        self.lenght.insert(0, "1000")
        self.lenght.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.lab_Icw = ctk.CTkLabel(self.input_frame, text="Ia; Ib; Ic [kA]")
        self.lab_Icw.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")
        self.Icw_txt = ctk.CTkEntry(self.input_frame)
        self.Icw_txt.insert(0, "187; -90; -90")
        self.Icw_txt.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        # Summary Frame
        self.summary_frame = ctk.CTkFrame(self.container_frame)
        self.summary_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.summary_frame.grid_columnconfigure(0, weight=1)

        self.desc_L = ctk.CTkLabel(self.summary_frame, text="length: 1000 [mm]")
        self.desc_L.grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.desc_IcwA = ctk.CTkLabel(self.summary_frame, text="Ia: 187.00 [kA]")
        self.desc_IcwA.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.desc_IcwB = ctk.CTkLabel(self.summary_frame, text="Ib: -90.00 [kA]")
        self.desc_IcwB.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self.desc_IcwC = ctk.CTkLabel(self.summary_frame, text="Ic: -90.00 [kA]")
        self.desc_IcwC.grid(row=3, column=0, padx=10, pady=(2, 10), sticky="w")
        
        # Console and Buttons Frame
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)

        self.tx1 = ctk.CTkTextbox(self.bottom_frame, height=120)
        self.tx1.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="nsew")

        self.rButton = ctk.CTkButton(
            self.bottom_frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        self.openButton = ctk.CTkButton(
            self.bottom_frame, text="Calculate Force Vectors", command=self.forcesAnalysis
        )
        self.openButton.grid(row=1, column=1, padx=10, pady=(5, 10), sticky="ew")

        # Set initial values
        self.readSettings()

    def readSettings(self):
        self.L = float(self.lenght.get())
        self.Icw = self.Icw_txt.get().split(";")
        self.Icw = [float(x) for x in self.Icw]

        self.desc_L.configure(text="length: {:.0f} [mm]".format(self.L))
        self.desc_IcwA.configure(text="Ia: {0[0]:.2f} [kA]".format(self.Icw))
        self.desc_IcwB.configure(text="Ib: {0[1]:.2f} [kA]".format(self.Icw))
        self.desc_IcwC.configure(text="Ic: {0[2]:.2f} [kA]".format(self.Icw))

        self.vPhA = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=1, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhB = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=2, dXmm=self.dXmm, dYmm=self.dYmm
        )
        self.vPhC = csd.n_arrayVectorize(
            inputArray=self.XsecArr, phaseNumber=3, dXmm=self.dXmm, dYmm=self.dYmm
        )

        self.elementsVector = np.concatenate((self.vPhA, self.vPhB, self.vPhC), axis=0)

    def console(self, string):
        self.tx1.insert(ctk.END, str(string))
        self.tx1.insert(tk.END, "\n")
        self.tx1.see(tk.END)

    def forcesAnalysis(self):
        # reading input data frm gui
        self.readSettings()
        self.Fa, self.Fb, self.Fc, self.ForcesMag2, self.ForcesVec = csd.n_getForces(
            XsecArr=self.XsecArr,
            vPhA=self.vPhA,
            vPhB=self.vPhB,
            vPhC=self.vPhC,
            Ia=self.Icw[0] * 1e3,
            Ib=self.Icw[1] * 1e3,
            Ic=self.Icw[2] * 1e3,
            Lenght=self.L * 1e-3,
        )

        # Reversing the  Y component sign to make it more 'natural'
        self.Fa = v2(self.Fa[0], -self.Fa[1])
        self.Fb = v2(self.Fb[0], -self.Fb[1])
        self.Fc = v2(self.Fc[0], -self.Fc[1])

        # Preparing the force density plot matrix
        self.ForcesMag2 = [abs(x / (self.dXmm * self.dYmm)) for x in self.ForcesMag2]

        self.resultsArray = csd.n_recreateresultsArray(
            elementsVector=self.elementsVector,
            resultsVector=self.ForcesMag2,
            initialGeometryArray=self.XsecArr,
        )

        self.console("Electrodynamic Forces:")
        self.console("Fa(x,y):({0[0]:.0f},{0[1]:.0f})[N]".format(self.Fa))
        self.console("Fb(x,y):({0[0]:.0f},{0[1]:.0f})[N]".format(self.Fb))
        self.console("Fc(x,y):({0[0]:.0f},{0[1]:.0f})[N]".format(self.Fc))

        print("Forces: \nA:{}\nB:{}\nC:{}".format(self.Fa, self.Fb, self.Fc))

        # Cecking the area in array that is used by geometry to limit the disp.
        min_row = int(np.min(self.elementsVector[:, 0]))
        max_row = int(np.max(self.elementsVector[:, 0]) + 1)

        min_col = int(np.min(self.elementsVector[:, 1]))
        max_col = int(np.max(self.elementsVector[:, 1]) + 1)

        self.resultsArray = self.resultsArray[min_row:max_row, min_col:max_col]

        plotWidth = (self.resultsArray.shape[1]) * self.dXmm
        plotHeight = (self.resultsArray.shape[0]) * self.dYmm

        fig = plt.figure("Forces Vectors")
        fig.clear()
        ax = plt.axes()

        my_cmap = matplotlib.cm.get_cmap("jet")
        my_cmap.set_under("w")

        im = ax.imshow(
            self.resultsArray,
            cmap=my_cmap,
            interpolation="none",
            vmin=0.8 * np.min(self.ForcesMag2),
            extent=[0, plotWidth, plotHeight, 0],
        )

        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            label="Force Density [N/mm$^2$]",
            alpha=0.5,
            fraction=0.046,
        )

        plt.axis("scaled")

        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2)
        position = list(csd.n_getPhasesCenters(self.vPhA, self.vPhB, self.vPhC))
        self.forces = [self.Fa, self.Fb, self.Fc]

        for k, p in enumerate(["A", "B", "C"]):
            if max_col - min_col >= max_row - min_row:
                x = position[k][0] - min_col * self.dXmm
                y = -(max_row - min_row) * self.dYmm
                ha = "center"
                va = "bottom"
                scale_units = "height"
                bigger_size = (max_col - min_col) * self.dXmm

            else:
                y = position[k][1] - min_row * self.dYmm
                x = -1.5 * (max_col - min_col) * self.dYmm
                ha = "right"
                va = "center"
                scale_units = "width"
                bigger_size = (max_row - min_row) * self.dYmm

            ax.text(
                x,
                y,
                "Phase {1}\n({0[0]:.0f},{0[1]:.0f})[N]".format(self.forces[k], p),
                ha=ha,
                va=va,
                rotation=0,
                size=10,
                bbox=bbox_props,
            )

        X = [position[i][0] - min_col * self.dXmm for i in range(3)]
        Y = [position[i][1] - min_row * self.dYmm for i in range(3)]

        U = [self.forces[i][0] for i in range(3)]
        V = [self.forces[i][1] for i in range(3)]

        maxForce = max([f.norm() for f in self.forces])

        plt.quiver(
            X,
            Y,
            U,
            V,
            edgecolor="none",
            facecolor="red",
            linewidth=0.5,
            scale=2 * maxForce,
            scale_units=scale_units,
            width=0.0001 * bigger_size,
        )

        conductors, total, phCon = csd.n_getConductors(
            XsecArr=self.XsecArr, vPhA=self.vPhA, vPhB=self.vPhB, vPhC=self.vPhC
        )

        bars = []

        for bar in range(1, total + 1):
            temp = csd.n_arrayVectorize(
                inputArray=conductors, phaseNumber=bar, dXmm=self.dXmm, dYmm=self.dYmm
            )
            bars.append(temp)

        Fx_array = [x[0] for x in self.ForcesVec]
        Fy_array = [-x[1] for x in self.ForcesVec]

        resultsFx = csd.n_recreateresultsArray(
            elementsVector=self.elementsVector,
            resultsVector=Fx_array,
            initialGeometryArray=self.XsecArr,
        )

        resultsFy = csd.n_recreateresultsArray(
            elementsVector=self.elementsVector,
            resultsVector=Fy_array,
            initialGeometryArray=self.XsecArr,
        )

        for i, bar in enumerate(bars):
            x, y = csd.n_getCenter(bar)
            x -= min_col * self.dXmm
            y -= min_row * self.dYmm

            ax.text(x, y, "[{}]".format(i), horizontalalignment="center")
            Fx = 0
            Fy = 0
            for element in bar:
                Fx += resultsFx[int(element[0]), int(element[1])]
                Fy += resultsFy[int(element[0]), int(element[1])]
            # Calculating bar perymiter - just for test nod needed in forces
            perymiter = csd.n_perymiter(bar, self.XsecArr, self.dXmm, self.dYmm)
            print(
                "Bar {0:02d}: F(x,y): ({1:06.2f}, {2:06.2f}) [N] pre: {3}".format(
                    i, Fx, Fy, perymiter
                )
            )

        plt.show()
