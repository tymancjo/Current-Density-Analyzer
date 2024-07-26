"""
This is a tkinter gui lib for CSD library and app
"""
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import math
import tkinter as tk
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2
from csdlib import csdos
import numpy as np

# matplotlib.use('TKAgg')


class MyPtrn:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)

        back = tk.Frame(master=self.top, width=300, height=5, bg="gray")
        back.pack()

        self.myLabel = tk.Label(top, text="Define the pattern parameters ")
        self.myLabel.pack()

        self.myLabel = tk.Label(top, text="Step in X (in cells): ")
        self.myLabel.pack()
        self.idX = tk.Entry(top, width=5)
        self.idX.pack()

        self.myLabel = tk.Label(top, text="Step in Y (in cells): ")
        self.myLabel.pack()
        self.idY = tk.Entry(top, width=5)
        self.idY.pack()

        self.myLabel = tk.Label(top, text="Number of copies: ")
        self.myLabel.pack()
        self.iN = tk.Entry(top, width=5)
        self.iN.pack()

        self.mySubmitButton = tk.Button(top, text="Submit", command=self.send)
        self.mySubmitButton.pack()

    def send(self):
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

        self.top.destroy()


class geometryModWindow:
    """
    This is a class that will be used to generate \
    the geometry modification Window
    """

    def __init__(self, master, canvas):
        self.master = master
        self.canvas = canvas
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.phase = tk.IntVar()

        self.phase.set(1)  # initialize

        tk.Radiobutton(
            master,
            text="Phase A",
            variable=self.phase,
            value=1,
            indicatoron=0,
            height=1,
            width=16,
            bg="red",
            highlightbackground="red",
        ).pack()
        tk.Radiobutton(
            master,
            text="Phase B",
            variable=self.phase,
            value=2,
            indicatoron=0,
            height=1,
            width=16,
            bg="green",
            highlightbackground="green",
        ).pack()
        tk.Radiobutton(
            master,
            text="Phase C",
            variable=self.phase,
            value=3,
            indicatoron=0,
            height=1,
            width=16,
            bg="blue",
            highlightbackground="blue",
        ).pack()

        self.aButton = tk.Button(self.frame, text="Click Me", command=self.shiftL)
        self.aButton.pack()
        self.XSecArray = super(geometryModWindow, self).XSecArray

    def shiftL(self):
        """This is just a zero argumet trigger for the geometry shift Button"""
        actualPhase = self.phase.get()
        csd.n_shiftPhase(actualPhase, -1, 0, XSecArray)
        print("Phase: {} shifed by {} x {}".format(actualPhase, dXmm, 0))
        csd.n_printTheArray(XSecArray, canvas=self.canvas)


class currentDensityWindowPro:
    """
    This class define the main control window for handling
    the analysis of current density of given geometry.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):

        self.getMaterials()
        print(self.Materials)

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm
        self.lenght = 1000
        self.CuGamma = 391.1  # [W/mK]
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.grid(row=0, column=0, ipadx=20, ipady=20)
        self.bframe = tk.Frame(self.master)
        self.bframe.grid(row=0, column=2, ipadx=20, ipady=20)

        self.cframe = tk.Frame(self.master)
        self.cframe.grid(row=0, column=3, ipadx=20, ipady=20)

        self.lab_I = tk.Label(self.frame, text="Current RMS [A]")
        self.lab_I.pack()
        self.Irms_txt = tk.Entry(self.frame, width=30)
        self.Irms_txt.insert(4, "1000;000;1000;120;1000;240")
        self.Irms_txt.pack()

        self.lab_Freq = tk.Label(self.frame, text="Frequency [Hz]")
        self.lab_Freq.pack()
        self.Freq_txt = tk.Entry(self.frame)
        self.Freq_txt.insert(5, "50")
        self.Freq_txt.pack()

        self.lab_Temp = tk.Label(self.frame, text="Conductor temperature [degC]")
        self.lab_Temp.pack()
        self.Temp_txt = tk.Entry(self.frame)
        self.Temp_txt.insert(5, "140")
        self.Temp_txt.pack()

        self.lab_Sigma = tk.Label(
            self.frame, text="Material conductivity at 20degC [S/m]"
        )
        self.lab_Sigma.pack()
        self.Sigma_txt = tk.Entry(self.frame)
        self.Sigma_txt.insert(5, "56e6")
        self.Sigma_txt.pack()

        self.lab_temCoRe = tk.Label(
            self.frame, text="Material Temp coeff of resistance [1/K]"
        )
        self.lab_temCoRe.pack()
        self.temCoRe_txt = tk.Entry(self.frame)
        self.temCoRe_txt.insert(5, "3.9e-3")
        self.temCoRe_txt.pack()

        self.lab_HTC = tk.Label(self.frame, text="HTC [W/m2K]")
        self.lab_HTC.pack()
        self.HTC_txt = tk.Entry(self.frame)
        self.HTC_txt.insert(5, "7")
        self.HTC_txt.pack()

        self.lab_ro = tk.Label(self.frame, text="material density [kg/3]")
        self.lab_ro.pack()
        self.ro_txt = tk.Entry(self.frame)
        self.ro_txt.insert(5, "8960")
        self.ro_txt.pack()

        self.lab_cp = tk.Label(self.frame, text="material heat capacity [J/kg.K]")
        self.lab_cp.pack()
        self.cp_txt = tk.Entry(self.frame)
        self.cp_txt.insert(5, "385")
        self.cp_txt.pack()

        self.lab_Gcon = tk.Label(self.frame, text="Thermal Conductivity [W/mK]")
        self.lab_Gcon.pack()
        self.Gcon_txt = tk.Entry(self.frame)
        self.Gcon_txt.insert(5, "0.25")
        self.Gcon_txt.pack()

        self.lab_Gmx = tk.Label(self.frame, text="Thermal Conductance Coef. Matrix")
        self.lab_Gmx.pack()
        self.Gmx_txt = tk.Entry(self.frame)
        self.Gmx_txt.insert(5, "0;0;0|0;0;0|0;0;0")
        self.Gmx_txt.pack()
        self.lab_Gmx = tk.Label(self.frame, text="Fa;Fab;Fac|Fba;Fb;Fbc|Fca;Fcb;Fc")
        self.lab_Gmx.pack()

        self.lab_lenght = tk.Label(self.frame, text="lenght [mm]")
        self.lab_lenght.pack()
        self.lenght_txt = tk.Entry(self.frame)
        self.lenght_txt.insert(5, "1000")
        self.lenght_txt.pack()

        self.rButton = tk.Button(
            self.frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.pack()

        #  reading the above entered stuff to variables

        self.I = self.Irms_txt.get().split(";")  # reading I as array
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())
        self.HTC = float(self.HTC_txt.get())
        self.Gcon = float(self.Gcon_txt.get())
        self.sigma20C = float(self.Sigma_txt.get())
        self.temCoRe = float(self.temCoRe_txt.get())
        self.ro = float(self.ro_txt.get())
        self.cp = float(self.cp_txt.get())

        self.Gmx = np.asarray(
            [gx.split(";") for gx in (self.Gmx_txt.get().split("|"))], dtype=float
        )

        self.desc_Ia = tk.Label(self.bframe, text=f"Ia: {self.I[0]} [A]")
        self.desc_Ia.pack()
        self.desc_Ib = tk.Label(self.bframe, text=f"Ib: {self.I[2]} [A]")
        self.desc_Ib.pack()
        self.desc_Ic = tk.Label(self.bframe, text=f"Ic: {self.I[4]} [A]")
        self.desc_Ic.pack()

        self.desc_f = tk.Label(
            self.bframe, text="Frequency: {:.2f} [Hz]".format(self.f)
        )
        self.desc_f.pack()
        self.desc_t = tk.Label(
            self.bframe, text="Temperature: {:.2f} [degC]".format(self.t)
        )
        self.desc_t.pack()

        self.desc_Sigma = tk.Label(
            self.bframe, text="Sigma: {:.2f} [MS/m]".format(self.sigma20C / 1000000)
        )
        self.desc_Sigma.pack()

        self.desc_temCoRe = tk.Label(
            self.bframe, text="alpha: {:.2f}e-3 [1/K]".format(self.temCoRe * 1000)
        )
        self.desc_temCoRe.pack()

        self.desc_ro = tk.Label(self.bframe, text="ro: {:.2f} [kg/m3]".format(self.ro))
        self.desc_ro.pack()

        self.desc_cp = tk.Label(self.bframe, text="cp: {:.2f} [J/kg.K]".format(self.cp))
        self.desc_cp.pack()

        self.desc_htc = tk.Label(
            self.bframe, text="HTC: {:.2f} [W/m2K]".format(self.HTC)
        )
        self.desc_htc.pack()
        self.desc_Gcon = tk.Label(
            self.bframe, text="Thermal Cond.: {:.2f} [W/mK]".format(self.Gcon)
        )
        self.desc_Gcon.pack()

        self.desc_lenght = tk.Label(
            self.bframe, text="lenght: {:.2f} [mm]".format(self.lenght)
        )
        self.desc_lenght.pack()

        self.tx1 = tk.Text(self.bframe, height=10, width=45)
        self.tx1.pack()

        self.openButton = tk.Button(
            self.bframe, text="Calculate!", command=self.powerAnalysis
        )
        self.openButton.pack()
        self.resultsButton = tk.Button(
            self.bframe, text="Recalulate Temp Rises", command=self.calcTempRise
        )
        self.resultsButton.pack()

        self.resultsButton = tk.Button(
            self.bframe, text="Show Results", command=self.showResults
        )
        self.resultsButton.pack()

        self.isSolved = False

        if self.Materials:
            self.material_buttons =[]
            for M in self.Materials:
                self.material_buttons.append(tk.Button(self.cframe, text=M.name, command=partial(self.setMaterial, M)))
                self.material_buttons[-1].pack()
                


        self.readSettings()

    def setMaterial(self,M):

        self.Sigma_txt.delete(0, tk.END)
        self.Sigma_txt.insert(0, M.sigma)

        self.temCoRe_txt.delete(0, tk.END)
        self.temCoRe_txt.insert(0, M.alpha)

        self.ro_txt.delete(0, tk.END)
        self.ro_txt.insert(0,M.ro)
        
        self.cp_txt.delete(0, tk.END)
        self.cp_txt.insert(0,M.cp)

        self.readSettings()


    def readSettings(self):
        self.I = self.Irms_txt.get().split(";")  # reading I as array
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

        self.desc_Ia.config(text=f"Ia: {self.I[0]} [A] at {self.I[1]} deg")
        self.desc_Ib.config(text=f"Ib: {self.I[2]} [A] at {self.I[3]} deg")
        self.desc_Ic.config(text=f"Ic: {self.I[4]} [A] at {self.I[5]} deg")

        self.desc_f.config(text="Frequency: {:.2f} [Hz]".format(self.f))
        self.desc_t.config(text="Temperature: {:.2f} [degC]".format(self.t))

        self.desc_Sigma.config(
            text="Sigma: {:.2f} [MS/m]".format(self.sigma20C / 1000000)
        )

        self.desc_temCoRe.config(
            text="alpha: {:.2f}e-3 [1/K]".format(self.temCoRe * 1000)
        )
        self.desc_ro.config(text="ro: {:.2f} [kg/m3]".format(self.ro))

        self.desc_cp.config(text="cp: {:.2f} [J/kg.K]".format(self.cp))

        self.desc_htc.config(text="HTC: {:.2f} [W/m2K]".format(self.HTC))
        self.desc_Gcon.config(text="Thermal Cond.: {:.5f} [W/mK]".format(self.Gcon))
        self.desc_lenght.config(text="lenght: {:.2f} [mm]".format(self.lenght))

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
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END, "\n")
        self.tx1.see(tk.END)

    def getMaterials(self):
        self.Materials = False
        list = csdos.read_file_to_list("materials.txt")[1:]
        print(list)
        if list:
            self.Materials = csdos.get_material_from_list(list)

    def powerAnalysis(self):
        self.readSettings()

        admitanceMatrix = np.linalg.inv(
            csd.n_getImpedanceArray(
                csd.n_getDistancesArray(self.elementsVector),
                freq=self.f,
                dXmm=self.dXmm,
                dYmm=self.dYmm,
                temperature=self.t,
                lenght=self.lenght,
                sigma20C=self.sigma20C,
                temCoRe=self.temCoRe,
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

        # Bringin each phase current to the assumer Irms level
        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)

        # expected Ia Ib Ic as symmetrical ones
        exIa = self.in_Ia
        exIb = self.in_Ib
        exIc = self.in_Ic

        # print('***VOLTAGES****')
        # print(Ua, Ub, Uc)

        # ratios of currents will give us new voltages for phases
        Ua = Ua * (exIa / Ia)
        Ub = Ub * (exIb / Ib)
        Uc = Uc * (exIc / Ic)

        # for debug:
        print("***recalculated votages****")
        print(Ua, Ub, Uc)
        print("***XXXXX****")

        # So we have now new volatges, lets solve again with them
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

        # Bringin each phase current to the assumer Irms level
        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)

        # end of second solve!

        # for debug:
        print("***pre calibration current results****")
        print(Ia, Ib, Ic)
        print(Ia + Ib + Ic)
        print("***XXXXX****")

        # Now we normalize up to the expecter self.I - just a polish
        # as we are almost there with the previous second solve for new VOLTAGES

        modIa = np.abs(Ia)
        modIb = np.abs(Ib)
        modIc = np.abs(Ic)

        # for debug:
        # print(modIa, modIb, modIc)

        currentPhA *= self.in_Ia / modIa
        currentPhB *= self.in_Ib / modIb
        currentPhC *= self.in_Ic / modIc

        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)

        getMod = np.vectorize(csd.n_getComplexModule)

        resultsCurrentVector = np.concatenate(
            (currentPhA, currentPhB, currentPhC), axis=0
        )
        # for debug
        # print(resultsCurrentVector)
        #
        resultsCurrentVector = getMod(resultsCurrentVector)
        resistanceVector = csd.n_getResistanceArray(
            self.elementsVector,
            dXmm=self.dXmm,
            dYmm=self.dYmm,
            temperature=self.t,
            lenght=self.lenght,
            sigma20C=self.sigma20C,
            temCoRe=self.temCoRe,
        )

        # This is the total power losses vector
        powerLossesVector = resistanceVector * resultsCurrentVector**2
        # This are the total power losses
        powerLosses = np.sum(powerLossesVector)

        # Power losses per phase
        powPhA = np.sum(powerLossesVector[0 : self.elementsPhaseA])
        powPhB = np.sum(
            powerLossesVector[
                self.elementsPhaseA : self.elementsPhaseA + self.elementsPhaseB : 1
            ]
        )
        powPhC = np.sum(powerLossesVector[self.elementsPhaseA + self.elementsPhaseB :])

        self.console(
            "power losses: {:.2f} [W] \n phA: {:.2f}[W]\n phB: {:.2f}[W]\n phC: {:.2f}[W]".format(
                powerLosses, powPhA, powPhB, powPhC
            )
        )

        self.powerLosses = [powerLosses, powPhA, powPhB, powPhC]

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
        self.resultsCurrentVector = resultsCurrentVector / (self.dXmm * self.dYmm)

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
            thGinv = np.linalg.inv(thG)
            thT = np.matmul(thGinv, thQ)

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

                DT = "[{}]\n1s: {:.2f}\n 3s: {:.2f}".format(
                    i, self.barsData[i][8] + 35, self.barsData[i][9] + 35
                )

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
                + "[mm] \n"
                "Ia:{:.1f}A {:.0f}$^o$ ".format(
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


class zWindow:
    """
    This class define the main control window for handling
    the analysis of equivalent phase impedance of given geometry.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):
        self.getMaterials()
        print(self.Materials)

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.grid(row=0,column=0)

        self.bframe = tk.Frame(self.master)
        self.bframe.grid(row=0, column=1)

        self.cframe = tk.Frame(self.master)
        self.cframe.grid(row=1,column=1)

        self.lab_Freq = tk.Label(self.frame, text="Frequency [Hz]")
        self.lab_Freq.pack()
        self.Freq_txt = tk.Entry(self.frame)
        self.Freq_txt.insert(5, "50")
        self.Freq_txt.pack()

        self.lab_Temp = tk.Label(self.frame, text="Conductor temperature [degC]")
        self.lab_Temp.pack()
        self.Temp_txt = tk.Entry(self.frame)
        self.Temp_txt.insert(5, "140")
        self.Temp_txt.pack()

        self.rButton = tk.Button(
            self.frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.pack()


        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f = tk.Label(
            self.bframe, text="Frequency: {:.2f} [Hz]".format(self.f)
        )
        self.desc_f.pack()
        self.desc_t = tk.Label(
            self.bframe, text="Temperature: {:.2f} [degC]".format(self.t)
        )
        self.desc_t.pack()


        self.tx1 = tk.Text(self.cframe, height=10, width=45)
        self.tx1.pack()

        self.openButton = tk.Button(
            self.cframe, text="Calculate!", command=self.powerAnalysis
        )
        self.openButton.pack()

        if self.Materials:
            self.material_buttons =[]
            for M in self.Materials:
                self.material_buttons.append(tk.Button(self.frame, text=M.name, command=partial(self.setMaterial, M)))
                self.material_buttons[-1].pack()

            self.M = self.Materials[0]


    def readSettings(self):
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f.config(text="Frequency: {:.2f} [Hz]".format(self.f))
        self.desc_t.config(text="Temperature: {:.2f} [degC]".format(self.t))

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
            list = csdos.read_file_to_list("materials.txt")[1:]
            print(list)
            if list:
                self.Materials = csdos.get_material_from_list(list)

    def setMaterial(self,M):
        self.M = M
        self.console(f"Set to use: {M.name} as material")

    def console(self, string):
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END, "\n")
        self.tx1.see(tk.END)

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


class zWindow3f:
    """
    This class define the main control window for handling
    the analysis of equivalent phase impedance of given geometry in case of 3f shunted scenario.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):

        self.getMaterials()
        print(self.Materials)

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.grid(row=0,column=0)

        self.bframe = tk.Frame(self.master)
        self.bframe.grid(row=0, column=1)

        self.cframe = tk.Frame(self.master)
        self.cframe.grid(row=1,column=1)

        self.lab_Freq = tk.Label(self.frame, text="Frequency [Hz]")
        self.lab_Freq.pack()
        self.Freq_txt = tk.Entry(self.frame)
        self.Freq_txt.insert(5, "50")
        self.Freq_txt.pack()

        self.lab_Temp = tk.Label(self.frame, text="Conductor temperature [degC]")
        self.lab_Temp.pack()
        self.Temp_txt = tk.Entry(self.frame)
        self.Temp_txt.insert(5, "140")
        self.Temp_txt.pack()

        self.rButton = tk.Button(
            self.frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.pack()


        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f = tk.Label(
            self.bframe, text="Frequency: {:.2f} [Hz]".format(self.f)
        )
        self.desc_f.pack()
        self.desc_t = tk.Label(
            self.bframe, text="Temperature: {:.2f} [degC]".format(self.t)
        )
        self.desc_t.pack()


        self.tx1 = tk.Text(self.cframe, height=10, width=45)
        self.tx1.pack()

        self.openButton = tk.Button(
            self.cframe, text="Calculate!", command=self.powerAnalysis
        )
        self.openButton.pack()

        if self.Materials:
            self.material_buttons =[]
            for M in self.Materials:
                self.material_buttons.append(tk.Button(self.frame, text=M.name, command=partial(self.setMaterial, M)))
                self.material_buttons[-1].pack()

            self.M = self.Materials[0]

    def readSettings(self):
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f.config(text="Frequency: {:.2f} [Hz]".format(self.f))
        self.desc_t.config(text="Temperature: {:.2f} [degC]".format(self.t))

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
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END, "\n")
        self.tx1.see(tk.END)
        
        


    def getMaterials(self):
            self.Materials = False
            list = csdos.read_file_to_list("materials.txt")[1:]
            print(list)
            if list:
                self.Materials = csdos.get_material_from_list(list)

    def setMaterial(self,M):
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


class forceWindow:
    """
    This class define the main control window for the
    electrodynamic forces analysis.
    """

    def __init__(self, master, XsecArr, dXmm, dYmm):
        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.lab_l = tk.Label(self.frame, text="Analysis lenght [mm]")
        self.lab_l.pack()
        self.lenght = tk.Entry(self.frame)
        self.lenght.insert(4, 1000)
        self.lenght.pack()

        self.lab_Icw = tk.Label(self.frame, text="Ia; Ib; Ic [kA]")
        self.lab_Icw.pack()
        self.Icw_txt = tk.Entry(self.frame)
        self.Icw_txt.insert(5, "187; -90; -90")
        self.Icw_txt.pack()

        self.rButton = tk.Button(
            self.frame, text="Set Parameters", command=self.readSettings
        )
        self.rButton.pack()

        self.Icw = self.Icw_txt.get().split(";")
        self.Icw = [float(x) for x in self.Icw]

        self.bframe = tk.Frame(self.master)
        self.bframe.pack(padx=10, pady=10)

        self.L = float(self.lenght.get())

        self.IcwA = "Ia: {0[0]:.2f} [kA]".format(self.Icw)
        self.IcwB = "Ia: {0[1]:.2f} [kA]".format(self.Icw)
        self.IcwC = "Ia: {0[2]:.2f} [kA]".format(self.Icw)

        self.desc_L = tk.Label(self.bframe, text="lenght: {:.0f} [mm]".format(self.L))
        self.desc_L.pack()
        self.desc_IcwA = tk.Label(self.bframe, text=self.IcwA)
        self.desc_IcwA.pack()
        self.desc_IcwB = tk.Label(self.bframe, text=self.IcwB)
        self.desc_IcwB.pack()
        self.desc_IcwC = tk.Label(self.bframe, text=self.IcwC)
        self.desc_IcwC.pack()

        self.cframe = tk.Frame(self.master)
        self.cframe.pack(padx=10, pady=10)

        self.tx1 = tk.Text(self.cframe, height=5, width=35)
        self.tx1.pack()

        self.openButton = tk.Button(
            self.cframe, text="Calculate Force Vectors", command=self.forcesAnalysis
        )
        self.openButton.pack()

    def readSettings(self):
        self.L = float(self.lenght.get())
        self.Icw = self.Icw_txt.get().split(";")
        self.Icw = [float(x) for x in self.Icw]

        self.desc_L.config(text="lenght: {:.0f} [mm]".format(self.L))
        self.desc_IcwA.config(text="Ia: {0[0]:.2f} [kA]".format(self.Icw))
        self.desc_IcwB.config(text="Ib: {0[1]:.2f} [kA]".format(self.Icw))
        self.desc_IcwC.config(text="Ic: {0[2]:.2f} [kA]".format(self.Icw))

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
        self.tx1.insert(tk.END, str(string))
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
