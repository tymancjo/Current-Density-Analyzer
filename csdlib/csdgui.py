'''
This is a tkinter gui lib for CSD library and app
'''

import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2
import numpy as np
# matplotlib.use('TKAgg')


class currentDensityWindow():
    '''
    This class define the main control window for handling
    the analysis of current density of given geometry.
    '''
    def __init__(self, master, XsecArr, dXmm, dYmm):

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm
        self.lenght = 1000
        self.CuGamma = 391.1  # [W/mK]
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.lab_I = tk.Label(self.frame,
                              text='Current RMS [A]')
        self.lab_I.pack()
        self.Irms_txt = tk.Entry(self.frame)
        self.Irms_txt.insert(4, 1000)
        self.Irms_txt.pack()

        self.lab_Freq = tk.Label(self.frame,
                                 text='Frequency [Hz]')
        self.lab_Freq.pack()
        self.Freq_txt = tk.Entry(self.frame)
        self.Freq_txt.insert(5, '50')
        self.Freq_txt.pack()

        self.lab_Temp = tk.Label(self.frame,
                                 text='Conductor temperature [degC]')
        self.lab_Temp.pack()
        self.Temp_txt = tk.Entry(self.frame)
        self.Temp_txt.insert(5, '140')
        self.Temp_txt.pack()

        self.lab_HTC = tk.Label(self.frame,
                                 text='HTC [W/m2K]')
        self.lab_HTC.pack()
        self.HTC_txt = tk.Entry(self.frame)
        self.HTC_txt.insert(5, '7')
        self.HTC_txt.pack()

        self.lab_lenght = tk.Label(self.frame,
                                 text='lenght [mm]')
        self.lab_lenght.pack()
        self.lenght_txt = tk.Entry(self.frame)
        self.lenght_txt.insert(5, '1000')
        self.lenght_txt.pack()

        self.rButton = tk.Button(self.frame, text='Set Parameters',
                                 command=self.readSettings)
        self.rButton.pack()

        self.bframe = tk.Frame(self.master)
        self.bframe.pack(padx=10, pady=10)

        self.I = float(self.Irms_txt.get())
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())
        self.HTC = float(self.HTC_txt.get())


        self.desc_I = tk.Label(self.bframe,
                               text='Current: {:.2f} [A]'.format(self.I))
        self.desc_I.pack()
        self.desc_f = tk.Label(self.bframe,
                               text='Frequency: {:.2f} [Hz]'.format(self.f))
        self.desc_f.pack()
        self.desc_t = tk.Label(self.bframe,
                               text='Temperature: {:.2f} [degC]'
                               .format(self.t))
        self.desc_t.pack()

        self.desc_htc = tk.Label(self.bframe,
                                       text='HTC: {:.2f} [W/m2K]'
                                       .format(self.HTC))
        self.desc_htc.pack()

        self.desc_lenght = tk.Label(self.bframe,
                                       text='lenght: {:.2f} [mm]'
                                       .format(self.lenght))
        self.desc_lenght.pack()

        self.cframe = tk.Frame(self.master)
        self.cframe.pack(padx=10, pady=10)

        self.tx1 = tk.Text(self.cframe, height=10, width=45)
        self.tx1.pack()

        self.openButton = tk.Button(self.cframe,
                                    text='Calculate!',
                                    command=self.powerAnalysis)
        self.openButton.pack()
        self.resultsButton = tk.Button(self.cframe,
                                       text='Show Results',
                                       command=self.showResults)
        self.resultsButton.pack()


    def readSettings(self):
        self.I = float(self.Irms_txt.get())
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())
        self.HTC = float(self.HTC_txt.get())
        self.lenght = float(self.lenght_txt.get())

        self.desc_I.config(text='Current: {:.2f} [A]'.format(self.I))
        self.desc_f.config(text='Frequency: {:.2f} [Hz]'.format(self.f))
        self.desc_t.config(text='Temperature: {:.2f} [degC]'.format(self.t))
        self.desc_htc.config(text='HTC: {:.2f} [W/m2K]'.format(self.HTC))
        self.desc_lenght.config(text='lenght: {:.2f} [mm]'.format(self.lenght))

        self.vPhA = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=1,
                                         dXmm=self.dXmm, dYmm=self.dYmm)
        self.vPhB = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=2,
                                         dXmm=self.dXmm, dYmm=self.dYmm)
        self.vPhC = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=3,
                                         dXmm=self.dXmm, dYmm=self.dYmm)

        # Lets put the all phases together
        self.elementsPhaseA = len(self.vPhA)
        self.elementsPhaseB = len(self.vPhB)
        self.elementsPhaseC = len(self.vPhC)

        if self.elementsPhaseA != 0 and self.elementsPhaseB != 0 and self.elementsPhaseC != 0:
            self.elementsVector = np.concatenate((self.vPhA,
                                                  self.vPhB,
                                                  self.vPhC),
                                                 axis=0)
        elif self.elementsPhaseA == 0:
            if self.elementsPhaseB == 0:
                self.elementsVector = self.vPhC
            elif self.elementsPhaseC == 0:
                self.elementsVector = self.vPhB
            else:
                self.elementsVector = np.concatenate((self.vPhB, self.vPhC),
                                                     axis=0)
        else:
            if self.elementsPhaseB == 0 and self.elementsPhaseC == 0:
                self.elementsVector = self.vPhA
            elif self.elementsPhaseC == 0:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhB),
                                                     axis=0)
            else:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhC),
                                                     axis=0)

    def console(self, string):
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END, '\n')
        self.tx1.see(tk.END)

    def powerAnalysis(self):
        self.readSettings()

        admitanceMatrix = np.linalg.inv(
                            csd.n_getImpedanceArray(
                                csd.n_getDistancesArray(self.elementsVector),
                                freq=self.f,
                                dXmm=self.dXmm,
                                dYmm=self.dYmm,
                                temperature=self.t,
                                lenght=self.lenght))

        # Let's put here some voltage vector
        Ua = complex(1, 0)
        Ub = complex(-0.5, np.sqrt(3)/2)
        Uc = complex(-0.5, -np.sqrt(3)/2)


        vA = np.ones(self.elementsPhaseA) * Ua
        vB = np.ones(self.elementsPhaseB) * Ub
        vC = np.ones(self.elementsPhaseC) * Uc


        voltageVector = np.concatenate((vA, vB, vC), axis=0)

        # Initial solve
        # Main equation solve
        currentVector = np.matmul(admitanceMatrix, voltageVector)

        # And now we need to get solution for each phase to normalize it
        currentPhA = currentVector[0: self.elementsPhaseA]
        currentPhB = currentVector[self.elementsPhaseA: self.elementsPhaseA + self.elementsPhaseB]
        currentPhC = currentVector[self.elementsPhaseA + self.elementsPhaseB:]

        # Bringin each phase current to the assumer Irms level
        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)

        # expected Ia Ib Ic as symmetrical ones
        exIa = self.I * complex(1, 0)
        exIb = self.I * complex(-0.5, np.sqrt(3)/2)
        exIc = self.I * complex(-0.5, -np.sqrt(3)/2)

        # print('***VOLTAGES****')
        # print(Ua, Ub, Uc)

        # ratios of currents will give us new voltages for phases
        Ua = Ua * (exIa / Ia)
        Ub = Ub * (exIb / Ib)
        Uc = Uc * (exIc / Ic)

        # for debug:
        # print(Ua, Ub, Uc)
        # print('***XXXXX****')

        # So we have now new volatges, lets solve again with them
        vA = np.ones(self.elementsPhaseA) * Ua
        vB = np.ones(self.elementsPhaseB) * Ub
        vC = np.ones(self.elementsPhaseC) * Uc

        voltageVector = np.concatenate((vA, vB, vC), axis=0)

        # Initial solve
        # Main equation solve
        currentVector = np.matmul(admitanceMatrix, voltageVector)

        # And now we need to get solution for each phase to normalize it
        currentPhA = currentVector[0: self.elementsPhaseA]
        currentPhB = currentVector[self.elementsPhaseA: self.elementsPhaseA + self.elementsPhaseB]
        currentPhC = currentVector[self.elementsPhaseA + self.elementsPhaseB:]

        # Bringin each phase current to the assumer Irms level
        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)

        # end of second solve!


        # for debug:
        # print('***XXXXX****')
        # print(Ia, Ib, Ic)
        # print(Ia + Ib + Ic)
        # print('***XXXXX****')

        # Now we normalize up to the expecter self.I - just a polish
        # as we are almost there with the previous second solve for new VOLTAGES

        modIa = np.abs(Ia)
        modIb = np.abs(Ib)
        modIc = np.abs(Ic)

        # for debug:
        # print(modIa, modIb, modIc)

        currentPhA *= (self.I / modIa)
        currentPhB *= (self.I / modIb)
        currentPhC *= (self.I / modIc)

        Ia = np.sum(currentPhA)
        Ib = np.sum(currentPhB)
        Ic = np.sum(currentPhC)

        getMod = np.vectorize(csd.n_getComplexModule)

        resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)
        # for debug
        # print(resultsCurrentVector)
        #
        resultsCurrentVector = getMod(resultsCurrentVector)
        resistanceVector = csd.n_getResistanceArray(self.elementsVector,
                                                    dXmm=self.dXmm,
                                                    dYmm=self.dYmm,
                                                    temperature=self.t,
                                                    lenght=self.lenght)

        # This is the total power losses vector
        powerLossesVector = resistanceVector * resultsCurrentVector**2
        # This are the total power losses
        powerLosses = np.sum(powerLossesVector)

        # Power losses per phase
        powPhA = np.sum(powerLossesVector[0:self.elementsPhaseA])
        powPhB = np.sum(powerLossesVector[self.elementsPhaseA:self.elementsPhaseA+self.elementsPhaseB:1])
        powPhC = np.sum(powerLossesVector[self.elementsPhaseA+self.elementsPhaseB:])

        self.console('power losses: {:.2f} [W] \n phA: {:.2f}[W]\n phB: {:.2f}[W]\n phC: {:.2f}[W]'
                     .format(powerLosses, powPhA, powPhB, powPhC))

        self.powerLosses = [powerLosses, powPhA, powPhB, powPhC]

        # Doing analysis per bar
        # Checking for the pabrs - separate conductor detecton

        conductors, total, self.phCon = csd.n_getConductors(XsecArr=self.XsecArr,
                                                       vPhA=self.vPhA,
                                                       vPhB=self.vPhB,
                                                       vPhC=self.vPhC)
        # self.phCon is the list of number of conductors per phase
        print(self.phCon)

        # Going thru the detected bars and preparing the arrays for each of it
        self.bars = []

        for bar in range(1, total+1):
            temp = csd.n_arrayVectorize(inputArray=conductors,
                                        phaseNumber=bar,
                                        dXmm=self.dXmm, dYmm=self.dYmm)
            self.bars.append(temp)

        # Converting resutls to current density
        self.resultsCurrentVector = resultsCurrentVector / (self.dXmm * self.dYmm)

        # Recreating the solution to form of cross section array
        self.resultsArray = csd.n_recreateresultsArray(
                                      elementsVector=self.elementsVector,
                                      resultsVector=self.resultsCurrentVector,
                                      initialGeometryArray=self.XsecArr)
        self.powerResultsArray = csd.n_recreateresultsArray(
                                      elementsVector=self.elementsVector,
                                      resultsVector=powerLossesVector,
                                      initialGeometryArray=self.XsecArr)

        # Doing the power losses sums per each bar
        # Vector to keep all power losses per bar data and perymeter
        # size and temp rise by given HTC

        self.barsData = []

        for i, bar in enumerate(self.bars):
            BarPowerLoss = 0
            print(len(bar))

            for element in bar:
                BarPowerLoss += self.powerResultsArray[int(element[0]),
                                                       int(element[1])]

            # Calculating bar perymiter of the current bar

            perymiter = csd.n_perymiter(bar, self.XsecArr, self.dXmm, self.dYmm)

            DT = BarPowerLoss / (perymiter * 1e-3 * self.HTC)

            XS = len(bar) * self.dXmm * self.dYmm

            self.barsData.append([BarPowerLoss, perymiter, DT, XS])

            # barsData structure
            # 0 power losses
            # 1 perymeter
            # 2 classic model DT
            # 3 cross section
            # 4 Ghtc to air thermal conductance
            # 5 Gt 1/2lenght thermal conductance
            # 6 Q power losses value
            # 7 New Thermal model DT

            # printing data for each bar
            print('Bar {0:02d}; Power; {1:06.2f}; [W]; perymeter; {2}; \
                  [mm]; TempRise; {3:.1f}; [K]'
                  .format(i, BarPowerLoss,  perymiter, DT))

        # Lets work with barsData for themral model calculations

        # first lets prepare for us some thermal data for each bar
        for bar in self.barsData:
            # calculationg the bar Ghtc
            p = bar[1]*1e-3
            A = bar[3]*1e-6
            lng = self.lenght*1e-3

            Ghtc = p * lng * self.HTC  # thermal conductance to air
            Gt = A * self.CuGamma / lng  # thermal conductance to com
            Q = bar[0] * lng  # Power losses value at lenght

            bar.append(Q)
            bar.append(Ghtc)
            bar.append(Gt)
            # now self.barsData have all the needed info :)

        # self.phCon is the list of number of conductors per phase
        phaseBars = [self.barsData[:self.phCon[0]],
                     self.barsData[self.phCon[0]:self.phCon[0]+self.phCon[1]],
                     self.barsData[self.phCon[0]+self.phCon[1]:]]

        self.Tout = []  # Prepare list of resulting Temps

        for bars in phaseBars:  # This loops over phases
            b = len(bars)
            Q = []
            G = []

            for i in range(b+1):
                # power vector preparation
                if i < b:
                    Q.append(bars[i][4])
                else:
                    Q.append(0)

                Grow = []  # just to keep for the moment the row of G matrix

                for j in range(b+1):
                    if j == i and j < b:
                        Grow.append(bars[j][5] + 2 * bars[j][6])
                    elif j == b and i < b:
                        Grow.append(-2 * bars[i][6])
                    elif i == b and j < b:
                        Grow.append(2 * bars[j][6])
                    elif j == i and j == b:  # bottom last element in matrix
                        Gtemp = 0
                        for k in range(b):
                            Gtemp += bars[k][6]
                        Grow.append(-2 * Gtemp)
                    else:
                        Grow.append(0)

                G.append(Grow)

            Q = np.array(Q)
            G_1 = np.linalg.inv(np.array(G))
            T = np.matmul(G_1, Q)

            print('***')
            print(T)
            print('***')

            for x in range(b):
                self.Tout.append(T[x])

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
                                      initialGeometryArray=self.XsecArr)

        for i, temp in enumerate(self.Tout):
            self.barsData[i].append(temp)
            print('Bar {}: {:.2f}[K]'.format(i,temp))

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

        b_phA = (perymeterA - 2*a) / 2
        b_phB = (perymeterB - 2*a) / 2
        b_phC = (perymeterC - 2*a) / 2

        # calculating equivalent gamma in 20C - to get the same power losses in DC calculations RI^2
        gamma_phA = (1 + alfa*(self.t - 20)) * 1 * self.I**2 / (a*1e-3 * b_phA*1e-3 * powPhA)
        gamma_phB = (1 + alfa*(self.t - 20)) * 1 * self.I**2 / (a*1e-3 * b_phB*1e-3 * powPhB)
        gamma_phC = (1 + alfa*(self.t - 20)) *1 * self.I**2 / (a*1e-3 * b_phC*1e-3 * powPhC)


        print('Equivalent bars for DC based thermal analysis: \n')
        print('Eqivalent bar phA is: {}mm x {}mm at gamma: {}'.format(a,b_phA,gamma_phA))
        print('Eqivalent bar phB is: {}mm x {}mm at gamma: {}'.format(a,b_phB,gamma_phB))
        print('Eqivalent bar phC is: {}mm x {}mm at gamma: {}'.format(a,b_phC,gamma_phC))

        print('({},{},1000, gamma={})'.format(a,b_phA,gamma_phA))
        print('({},{},1000, gamma={})'.format(a,b_phB,gamma_phB))
        print('({},{},1000, gamma={})'.format(a,b_phC,gamma_phC))



        # Display the results:
        self.showResults()


    def showResults(self):

        title_font = { 'size':'11', 'color':'black', 'weight':'normal'}
        axis_font = { 'size':'10'}

        if np.sum(self.resultsArray) != 0:

            # Cecking the area in array that is used by geometry to limit the display
            min_row = int(np.min(self.elementsVector[:, 0]))
            max_row = int(np.max(self.elementsVector[:, 0])+1)

            min_col = int(np.min(self.elementsVector[:, 1]))
            max_col = int(np.max(self.elementsVector[:, 1])+1)

            # Cutting down results array to the area with geometry
            tempriseArrayDisplay = self.tempriseResultsArray[min_row:max_row, min_col:max_col]
            resultsArrayDisplay = self.resultsArray[min_row:max_row, min_col:max_col]

            # Checking out what are the dimensions od the ploted area
            # to make propper scaling

            plotWidth = (resultsArrayDisplay.shape[1]) * self.dXmm
            plotHeight = (resultsArrayDisplay.shape[0]) * self.dYmm

            fig = plt.figure('Power Results Window')
            ax = fig.add_subplot(1, 1, 1)

            my_cmap = matplotlib.cm.get_cmap('jet')
            my_cmap.set_under('w')

            im = ax.imshow(resultsArrayDisplay,
                           cmap=my_cmap, interpolation='none',
                           vmin=0.8*np.min(self.resultsCurrentVector),
                           extent=[0, plotWidth, plotHeight, 0])

            fig.colorbar(im, ax=ax, orientation='vertical',
                         label='Current Density [A/mm$^2$]',
                         alpha=0.5, fraction=0.046)
            plt.axis('scaled')

            # Putting the detected bars numvers on plot to reffer the console data
            # And doing calculation for each bar

            for i, bar in enumerate(self.bars):
                x, y = csd.n_getCenter(bar)
                x -= min_col * self.dXmm
                y -= min_row * self.dYmm

                ax.text(x, y, '[{}]'.format(i), horizontalalignment='center')
                self.console('bar {0:02d}: {1:.01f}[K]'.format(i, self.barsData[i][7]))

            # *** end of the per bar analysis ***

            ax.set_title(str(self.f)+'[Hz] / '+str(self.I)+'[A] / '+str(self.t) +
                         '[$^o$C] /'+str(self.lenght) +
                         '[mm]\n Power Losses {0[0]:.2f}[W] \n phA: {0[1]:.2f} phB: {0[2]:.2f} phC: {0[3]:.2f}'.format(self.powerLosses), **title_font)

            plt.xlabel('size [mm]', **axis_font)
            plt.ylabel('size [mm]', **axis_font)

            fig.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

            plt.tight_layout()

            self.showTemperatureResults()
            plt.show()


    def showTemperatureResults(self):

        title_font = { 'size':'11', 'color':'black', 'weight':'normal'}
        axis_font = { 'size':'10'}

        if np.sum(self.resultsArray) != 0:

            # Cecking the area in array that is used by geometry to limit the display
            min_row = int(np.min(self.elementsVector[:, 0]))
            max_row = int(np.max(self.elementsVector[:, 0])+1)

            min_col = int(np.min(self.elementsVector[:, 1]))
            max_col = int(np.max(self.elementsVector[:, 1])+1)

            # Cutting down results array to the area with geometry
            resultsArrayDisplay = self.tempriseResultsArray[min_row:max_row, min_col:max_col]

            # Checking out what are the dimensions od the ploted area
            # to make propper scaling

            plotWidth = (resultsArrayDisplay.shape[1]) * self.dXmm
            plotHeight = (resultsArrayDisplay.shape[0]) * self.dYmm

            fig = plt.figure('Temperature Results Window')
            ax = fig.add_subplot(1, 1, 1)

            my_cmap = matplotlib.cm.get_cmap('jet')
            my_cmap.set_under('w')

            im = ax.imshow(resultsArrayDisplay,
                           cmap=my_cmap, interpolation='none',
                           vmin=0.8*np.min(self.Tout),
                           extent=[0, plotWidth, plotHeight, 0])

            fig.colorbar(im, ax=ax, orientation='vertical',
                         label='Temperature Rise [K]',
                         alpha=0.5, fraction=0.046)
            plt.axis('scaled')

            # Putting the detected bars numvers on plot to reffer the console data
            # And doing calculation for each bar

            for i, bar in enumerate(self.bars):
                x, y = csd.n_getCenter(bar)
                x -= min_col * self.dXmm
                y -= min_row * self.dYmm

                ax.text(x, y, '[{}]'.format(i), horizontalalignment='center')

            # *** end of the per bar analysis ***

            ax.set_title(str(self.f)+'[Hz] / '+str(self.I)+'[A] / '+str(self.t) +
                         '[$^o$C] /'+str(self.lenght) +
                         '[mm]\n Temperature Rises in Bars', **title_font)

            plt.xlabel('size [mm]', **axis_font)
            plt.ylabel('size [mm]', **axis_font)

            fig.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

            plt.tight_layout()
            # plt.show()

class zWindow():
    '''
    This class define the main control window for handling
    the analysis of equivalent phase impedance of given geometry.
    '''
    def __init__(self, master, XsecArr, dXmm, dYmm):

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.lab_Freq = tk.Label(self.frame,
                                 text='Frequency [Hz]')
        self.lab_Freq.pack()
        self.Freq_txt = tk.Entry(self.frame)
        self.Freq_txt.insert(5, '50')
        self.Freq_txt.pack()

        self.lab_Temp = tk.Label(self.frame,
                                 text='Conductor temperature [degC]')
        self.lab_Temp.pack()
        self.Temp_txt = tk.Entry(self.frame)
        self.Temp_txt.insert(5, '140')
        self.Temp_txt.pack()

        self.rButton = tk.Button(self.frame, text='Set Parameters',
                                 command=self.readSettings)
        self.rButton.pack()

        self.bframe = tk.Frame(self.master)
        self.bframe.pack(padx=10, pady=10)

        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f = tk.Label(self.bframe,
                               text='Frequency: {:.2f} [Hz]'.format(self.f))
        self.desc_f.pack()
        self.desc_t = tk.Label(self.bframe,
                               text='Temperature: {:.2f} [degC]'
                               .format(self.t))
        self.desc_t.pack()

        self.cframe = tk.Frame(self.master)
        self.cframe.pack(padx=10, pady=10)

        self.tx1 = tk.Text(self.cframe, height=10, width=45)
        self.tx1.pack()

        self.openButton = tk.Button(self.cframe,
                                    text='Calculate!',
                                    command=self.powerAnalysis)
        self.openButton.pack()



    def readSettings(self):
        self.f = float(self.Freq_txt.get())
        self.t = float(self.Temp_txt.get())

        self.desc_f.config(text='Frequency: {:.2f} [Hz]'.format(self.f))
        self.desc_t.config(text='Temperature: {:.2f} [degC]'.format(self.t))

        self.vPhA = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=1,
                                         dXmm=self.dXmm, dYmm=self.dYmm)
        self.vPhB = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=2,
                                         dXmm=self.dXmm, dYmm=self.dYmm)
        self.vPhC = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=3,
                                         dXmm=self.dXmm, dYmm=self.dYmm)

        self.elementsPhaseA = len(self.vPhA)
        self.elementsPhaseB = len(self.vPhB)
        self.elementsPhaseC = len(self.vPhC)

        if self.elementsPhaseA != 0 and self.elementsPhaseB != 0 and self.elementsPhaseC != 0:
            self.elementsVector = np.concatenate((self.vPhA,
                                                  self.vPhB,
                                                  self.vPhC),
                                                 axis=0)
        elif self.elementsPhaseA == 0:
            if self.elementsPhaseB == 0:
                self.elementsVector = self.vPhC
            elif self.elementsPhaseC == 0:
                self.elementsVector = self.vPhB
            else:
                self.elementsVector = np.concatenate((self.vPhB, self.vPhC),
                                                     axis=0)
        else:
            if self.elementsPhaseB == 0 and self.elementsPhaseC == 0:
                self.elementsVector = self.vPhA
            elif self.elementsPhaseC == 0:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhB),
                                                     axis=0)
            else:
                self.elementsVector = np.concatenate((self.vPhA, self.vPhC),
                                                     axis=0)

    def console(self, string):
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END, '\n')
        self.tx1.see(tk.END)

    def powerAnalysis(self):
        self.readSettings()


        # Let's put here some voltage vector
        # initial voltage values
        Ua = complex(1, 0)
        Ub = complex(-0.5, np.sqrt(3)/2)
        Uc = complex(-0.5, -np.sqrt(3)/2)

        admitanceMatrix = np.linalg.inv(
        csd.n_getImpedanceArray(
        csd.n_getDistancesArray(self.vPhA),
        freq=self.f,
        dXmm=self.dXmm,
        dYmm=self.dYmm,
        temperature=self.t))


        voltageVector = np.ones(len(self.vPhA)) * Ua

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        Ia = np.sum(currentVector)

        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Za = Ua / Ia
        La = Za.imag / (2*np.pi*self.f)

        # round 2 - phase B - other phases shunted

        admitanceMatrix = np.linalg.inv(
        csd.n_getImpedanceArray(
        csd.n_getDistancesArray(self.vPhB),
        freq=self.f,
        dXmm=self.dXmm,
        dYmm=self.dYmm,
        temperature=self.t))


        voltageVector = np.ones(len(self.vPhB)) * Ub

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        Ib = np.sum(currentVector)


        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Zb = Ub / Ib
        Lb = Zb.imag / (2*np.pi*self.f)

        # round 3 - phase C - other phases shunted
        admitanceMatrix = np.linalg.inv(
        csd.n_getImpedanceArray(
        csd.n_getDistancesArray(self.vPhC),
        freq=self.f,
        dXmm=self.dXmm,
        dYmm=self.dYmm,
        temperature=self.t))


        voltageVector = np.ones(len(self.vPhC)) * Uc

        currentVector = np.matmul(admitanceMatrix, voltageVector)


        Ic = np.sum(currentVector)

        # As we have complex currents vectors we can caluculate the impedances
        # Of each phase as Z= U/I

        Zc = Uc / Ic
        Lc = Zc.imag / (2*np.pi*self.f)


        print('Impedance calulations results: \n')
        print('Za: {:.2f}  [uOhm]  La = {:.3f} [uH]'.format(Za * 1e6, La * 1e6))
        print('Zb: {:.2f}  [uOhm]  Lb = {:.3f} [uH]'.format(Zb * 1e6, Lb * 1e6))
        print('Zc: {:.2f}  [uOhm]  Lc = {:.3f} [uH]'.format(Zc * 1e6, Lc * 1e6))
        print('########################################################\n \n')


        # printing to GUI console window
        self.console('Impedance calulations results:')
        self.console('Za: {:.2f} [uOhm]'.format(Za * 1e6))
        self.console('Zb: {:.2f} [uOhm]'.format(Zb * 1e6))
        self.console('Zc: {:.2f} [uOhm]'.format(Zc * 1e6))
        self.console('')
        self.console('La: {:.3f} [uH]'.format(La * 1e6))
        self.console('Lb: {:.3f} [uH]'.format(Lb * 1e6))
        self.console('Lc: {:.3f} [uH]'.format(Lc * 1e6))


class forceWindow():
    '''
    This class define the main control window for the
    electrodynamic forces analysis.
    '''
    def __init__(self, master, XsecArr, dXmm, dYmm):

        self.XsecArr = XsecArr
        self.dXmm = dXmm
        self.dYmm = dYmm

        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        self.lab_l = tk.Label(self.frame, text='Analysis lenght [mm]')
        self.lab_l.pack()
        self.lenght = tk.Entry(self.frame)
        self.lenght.insert(4, 1000)
        self.lenght.pack()

        self.lab_Icw = tk.Label(self.frame, text='Ia; Ib; Ic [kA]')
        self.lab_Icw.pack()
        self.Icw_txt = tk.Entry(self.frame)
        self.Icw_txt.insert(5, '187; -90; -90')
        self.Icw_txt.pack()

        self.rButton = tk.Button(self.frame, text='Set Parameters',
                                 command=self.readSettings)
        self.rButton.pack()

        self.Icw = self.Icw_txt.get().split(';')
        self.Icw = [float(x) for x in self.Icw]

        self.bframe = tk.Frame(self.master)
        self.bframe.pack(padx=10, pady=10)

        self.L = float(self.lenght.get())

        self.IcwA = 'Ia: {0[0]:.2f} [kA]'.format(self.Icw)
        self.IcwB = 'Ia: {0[1]:.2f} [kA]'.format(self.Icw)
        self.IcwC = 'Ia: {0[2]:.2f} [kA]'.format(self.Icw)

        self.desc_L = tk.Label(self.bframe,
                               text='lenght: {:.0f} [mm]'.format(self.L))
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

        self.openButton = tk.Button(self.cframe,
                                    text='Calculate Force Vectors',
                                    command=self.forcesAnalysis)
        self.openButton.pack()

    def readSettings(self):
        self.L = float(self.lenght.get())
        self.Icw = self.Icw_txt.get().split(';')
        self.Icw = [float(x) for x in self.Icw]

        self.desc_L.config(text='lenght: {:.0f} [mm]'.format(self.L))
        self.desc_IcwA.config(text='Ia: {0[0]:.2f} [kA]'.format(self.Icw))
        self.desc_IcwB.config(text='Ib: {0[1]:.2f} [kA]'.format(self.Icw))
        self.desc_IcwC.config(text='Ic: {0[2]:.2f} [kA]'.format(self.Icw))

        self.vPhA = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=1,
                                         dXmm=self.dXmm, dYmm=self.dYmm)
        self.vPhB = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=2,
                                         dXmm=self.dXmm, dYmm=self.dYmm)
        self.vPhC = csd.n_arrayVectorize(inputArray=self.XsecArr,
                                         phaseNumber=3,
                                         dXmm=self.dXmm, dYmm=self.dYmm)

        self.elementsVector = np.concatenate((self.vPhA, self.vPhB, self.vPhC),
                                             axis=0)

    def console(self, string):
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END, '\n')
        self.tx1.see(tk.END)

    def forcesAnalysis(self):
        # reading input data frm gui
        self.readSettings()
        self.Fa, self.Fb, self.Fc, self.ForcesMag2,\
            self.ForcesVec = csd.n_getForces(XsecArr=self.XsecArr,
                                             vPhA=self.vPhA,
                                             vPhB=self.vPhB,
                                             vPhC=self.vPhC,
                                             Ia=self.Icw[0]*1e3,
                                             Ib=self.Icw[1]*1e3,
                                             Ic=self.Icw[2]*1e3,
                                             Lenght=self.L*1e-3)

        # Reversing the  Y component sign to make it more 'natural'
        self.Fa = v2(self.Fa[0], -self.Fa[1])
        self.Fb = v2(self.Fb[0], -self.Fb[1])
        self.Fc = v2(self.Fc[0], -self.Fc[1])

        # Preparing the force density plot matrix
        self.ForcesMag2 = [abs(x / (self.dXmm * self.dYmm))
                           for x in self.ForcesMag2]

        self.resultsArray =\
            csd.n_recreateresultsArray(elementsVector=self.elementsVector,
                                       resultsVector=self.ForcesMag2,
                                       initialGeometryArray=self.XsecArr)

        self.console('Electrodynamic Forces:')
        self.console('Fa(x,y):({0[0]:.0f},{0[1]:.0f})[N]'.format(self.Fa))
        self.console('Fb(x,y):({0[0]:.0f},{0[1]:.0f})[N]'.format(self.Fb))
        self.console('Fc(x,y):({0[0]:.0f},{0[1]:.0f})[N]'.format(self.Fc))

        print('Forces: \nA:{}\nB:{}\nC:{}'.format(self.Fa, self.Fb, self.Fc))

        # Cecking the area in array that is used by geometry to limit the disp.
        min_row = int(np.min(self.elementsVector[:, 0]))
        max_row = int(np.max(self.elementsVector[:, 0])+1)

        min_col = int(np.min(self.elementsVector[:, 1]))
        max_col = int(np.max(self.elementsVector[:, 1])+1)

        self.resultsArray = self.resultsArray[min_row: max_row,
                                              min_col: max_col]

        plotWidth = (self.resultsArray.shape[1]) * self.dXmm
        plotHeight = (self.resultsArray.shape[0]) * self.dYmm

        fig = plt.figure('Forces Vectors')
        fig.clear()
        ax = plt.axes()

        my_cmap = matplotlib.cm.get_cmap('jet')
        my_cmap.set_under('w')

        im = ax.imshow(self.resultsArray, cmap=my_cmap, interpolation='none',
                       vmin=0.8*np.min(self.ForcesMag2),
                       extent=[0, plotWidth, plotHeight, 0])

        fig.colorbar(im, ax=ax, orientation='vertical',
                     label='Force Density [N/mm$^2$]',
                     alpha=0.5, fraction=0.046)

        plt.axis('scaled')

        bbox_props = dict(boxstyle="round,pad=0.3",
                          fc="white", ec="black", lw=2)
        position = list(csd.n_getPhasesCenters(self.vPhA, self.vPhB,
                                               self.vPhC))
        self.forces = [self.Fa, self.Fb, self.Fc]

        for k, p in enumerate(['A', 'B', 'C']):

            if (max_col - min_col >= max_row - min_row):
                x = position[k][0] - min_col * self.dXmm
                y = -(max_row - min_row) * self.dYmm
                ha = "center"
                va = "bottom"
                scale_units = 'height'
                bigger_size = (max_col - min_col)*self.dXmm

            else:
                y = position[k][1] - min_row * self.dYmm
                x = -1.5 * (max_col - min_col) * self.dYmm
                ha = "right"
                va = "center"
                scale_units = 'width'
                bigger_size = (max_row - min_row)*self.dYmm

            ax.text(x, y,
                    "Phase {1}\n({0[0]:.0f},{0[1]:.0f})[N]"
                    .format(self.forces[k], p),
                    ha=ha, va=va,
                    rotation=0,
                    size=10,
                    bbox=bbox_props)

        X = [position[i][0] - min_col * self.dXmm for i in range(3)]
        Y = [position[i][1] - min_row * self.dYmm for i in range(3)]

        U = [self.forces[i][0] for i in range(3)]
        V = [self.forces[i][1] for i in range(3)]

        maxForce = max([f.norm() for f in self.forces])

        plt.quiver(X, Y, U, V, edgecolor='none', facecolor='red',
                   linewidth=.5, scale=2 * maxForce, scale_units=scale_units,
                   width=.0001 * bigger_size)

        conductors, total, phCon = csd.n_getConductors(XsecArr=self.XsecArr,
                                                       vPhA=self.vPhA,
                                                       vPhB=self.vPhB,
                                                       vPhC=self.vPhC)

        bars = []

        for bar in range(1, total+1):
            temp = csd.n_arrayVectorize(inputArray=conductors,
                                             phaseNumber=bar,
                                             dXmm=self.dXmm, dYmm=self.dYmm)
            bars.append(temp)



        Fx_array = [x[0] for x in self.ForcesVec]
        Fy_array = [-x[1] for x in self.ForcesVec]

        resultsFx =\
            csd.n_recreateresultsArray(elementsVector=self.elementsVector,
                                       resultsVector=Fx_array,
                                       initialGeometryArray=self.XsecArr)

        resultsFy =\
            csd.n_recreateresultsArray(elementsVector=self.elementsVector,
                                       resultsVector=Fy_array,
                                       initialGeometryArray=self.XsecArr)

        for i, bar in enumerate(bars):
            x, y = csd.n_getCenter(bar)
            ax.text(x, y, '[{}]'.format(i), horizontalalignment='center')
            Fx = 0
            Fy = 0
            for element in bar:
                Fx += resultsFx[int(element[0]), int(element[1])]
                Fy += resultsFy[int(element[0]), int(element[1])]
            # Calculating bar perymiter - just for test nod needed in forces
            perymiter = csd.n_perymiter(bar, self.XsecArr, self.dXmm, self.dYmm)
            print('Bar {0:02d}: F(x,y): ({1:06.2f}, {2:06.2f}) [N] pre: {3}'.format(i, Fx, Fy, perymiter))

        plt.show()
