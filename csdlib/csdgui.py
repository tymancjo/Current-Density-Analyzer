'''
This is a tkinter gui lib for CSD library and app
'''

import tkinter as tk
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2

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

        self.desc_L = tk.Label(self.bframe, text='lenght: {:.0f} [mm]'.format(self.L))
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

        self.openButton = tk.Button(self.cframe, text='Calculate Force Vectors',
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

    def console(self, string):
        self.tx1.insert(tk.END, str(string))
        self.tx1.insert(tk.END,'\n')
        self.tx1.see(tk.END)


    def forcesAnalysis(self):
        # reading input data frm gui
        self.readSettings()
        self.Fa, self.Fb, self.Fc = csd.n_getForces(XsecArr=self.XsecArr,
                                                    vPhA=self.vPhA,
                                                    vPhB=self.vPhB,
                                                    vPhC=self.vPhC,
                                                    Ia=self.Icw[0]*1e3,
                                                    Ib=self.Icw[1]*1e3,
                                                    Ic=self.Icw[2]*1e3,
                                                    Lenght=self.L*1e-3)

        self.console('Electrodynamic Forces:')
        self.console('Fa(x,y):({0[0]:.0f},{0[1]:.0f})[N]'.format(self.Fa))
        self.console('Fb(x,y):({0[0]:.0f},{0[1]:.0f})[N]'.format(self.Fb))
        self.console('Fc(x,y):({0[0]:.0f},{0[1]:.0f})[N]'.format(self.Fc))
        
        print('Forces: \nA:{}\nB:{}\nC:{}'.format(self.Fa, self.Fb, self.Fc))

        pa, pb, pc = csd.n_getPhasesCenters(self.vPhA, self.vPhB, self.vPhC)
        print(pa, pb, pc)

