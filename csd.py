import matplotlib
import matplotlib.pyplot as plt
from tkinter import * 
from tkinter import filedialog, messagebox
import numpy as np
import os.path
import time
# Importing local library
from csdlib import csdlib as csd
from csdlib.vect import Vector as v2
from csdlib import csdgui as gui

matplotlib.use('TKAgg')


def showXsecArray(event):
    '''
    This function print the array to the terminal
    '''
    print(XSecArray)


def saveArrayToFile():
    '''
    This function saves the data of cross section array to file
    '''
    filename = filedialog.asksaveasfilename()
    filename = os.path.normpath(filename)
    if filename:
        saveTheData(filename)


def saveTheData(filename):
    '''
    This is the subfunction for saving data
    '''
    print('Saving to file :' + filename)
    np.save(filename, XSecArray)


def loadArrayFromFile():
    '''
    This function loads the data from the file
    !!!!! Need some work - it dosn't reset properly the dXmm and dYmm
    '''
    filename = filedialog.askopenfilename()
    filename = os.path.normpath(filename)

    if np.sum(XSecArray) != 0: # Test if there is anything draw on the array
        q = messagebox.askquestion("Delete", "This will delete current shape. Are You Sure?", icon='warning')
        if q == 'yes':
            if filename:
                loadTheData(filename)
    else:
        if filename:
            loadTheData(filename)


def loadTheData(filename):
    '''
    This is sub function to load data
    '''
    global XSecArray
    print('Readinf from file :' + filename)
    XSecArray =  np.load(filename)
    csd.n_printTheArray(XSecArray, canvas=w)


def zoomInArray(inputArray, zoomSize=2, startX=0, startY=0):

    oryginalX = inputArray.shape[0]
    oryginalY = inputArray.shape[1]

    NewX = oryginalX // zoomSize
    NewY = oryginalY // zoomSize

    if startX > (oryginalX-NewX):
        startX = oryginalX-NewX

    if startY > (oryginalY-NewY):
        startY = oryginalY-NewY

    return inputArray[startY:startY+NewY,startX:startX+NewX]


def zoomIn():
    global globalZoom

    if globalZoom < 5:
        globalZoom +=1

    csd.n_printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY),
                        canvas=w)


def zoomOut():
    global globalZoom, globalX, globalY

    if globalZoom > 1:
        globalZoom -= 1

    globalX -= 2
    if globalX < 0:
        globalX = 0

    globalY -= 2
    if globalY < 0:
        globalY = 0

    if globalZoom == 1:
        globalX = 0
        globalY = 0

    csd.n_printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY),
                        canvas=w)


def zoomL():
    global globalX, globalY

    globalX -= 2
    if globalX < 0:
        globalX = 0
    csd.n_printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY),
                        canvas=w)   


def zoomR():
    global globalX, globalY

    globalX += 2

    if globalX > XSecArray.shape[1]-XSecArray.shape[1]//globalZoom:
        globalX = XSecArray.shape[1]-XSecArray.shape[1]//globalZoom

    csd.n_printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY),
                        canvas=w)


def zoomU():
    global globalX, globalY

    globalY -=2
    if globalY < 0:
        globalY =0

    csd.n_printTheArray(zoomInArray(XSecArray,globalZoom,globalX,globalY), canvas=w)


def zoomD():
    global globalX, globalY

    globalY +=2
    if globalY > XSecArray.shape[0]-XSecArray.shape[0]//globalZoom:
        globalY = XSecArray.shape[0]-XSecArray.shape[0]//globalZoom

    csd.n_printTheArray(zoomInArray(XSecArray,globalZoom,globalX,globalY), canvas=w)

def displayArrayAsImage():
    '''
    This function print the array to termianl and shows additional info of the
    dX and dy size in mm
    and redraw the array on the graphical working area
    '''
    print(XSecArray)
    print(str(dXmm)+'[mm] :'+str(dYmm)+'[mm]')
    csd.n_printTheArray(zoomInArray(XSecArray, globalZoom, globalX, globalY),
                        canvas=w)

    drawGeometryArray(XSecArray)


def setPoint(event):
    '''Trigger procesdure for GUI action'''
    actualPhase = phase.get()

    csd.n_setUpPoint(event, Set=actualPhase,
                     dataArray=zoomInArray(XSecArray, globalZoom, globalX,
                                           globalY), canvas=w)

    #  Plotting on CAD view if exist
    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass


def resetPoint(event):
    '''Trigger procesdure for GUI action'''
    csd.n_setUpPoint(event, Set=0, dataArray=zoomInArray(XSecArray,globalZoom,globalX,globalY), canvas=w)

    #  Plotting on CAD view if exist
    try:
        geomim.set_data(XSecArray)
        plt.draw()
    except:
        pass

def clearArrayAndDisplay():
    '''
    This function erase the datat form array and return it back to initial
    setup
    '''
    global XSecArray, dX, dY
    if np.sum(XSecArray) != 0: # Test if there is anything draw on the array
        q = messagebox.askquestion("Delete", "This will delete current shape. Are You Sure?", icon='warning')
        if q == 'yes':
            XSecArray = np.zeros(XSecArray.shape)
            #checkered(w, dX, dY)
            mainSetup()
            csd.n_checkered(w, elementsInX, elementsInY)
            myEntryDx.delete(0,END)
            myEntryDx.insert(END,str(dXmm))
            setParameters()

    else:
            XSecArray = np.zeros(XSecArray.shape)
            #checkered(w, dX, dY)
            mainSetup()
            csd.n_checkered(w, elementsInX, elementsInY)
            myEntryDx.delete(0,END)
            myEntryDx.insert(END,str(dXmm))
            setParameters()





def subdivideArray():
    '''
    This function is logical wrapper for array slicer
    it take care to not loose any entered data from the modufied array
    '''

    start= time.clock() #just to check the time

    global XSecArray, dXmm, dYmm
    if dXmm > 1 and dYmm > 1:
        XSecArray = csd.n_arraySlicer(inputArray = XSecArray, subDivisions = 2)

        dXmm = dXmm/2
        dYmm = dYmm/2

        print(str(dXmm)+'[mm] :'+str(dYmm)+'[mm]')
        csd.n_printTheArray(dataArray=XSecArray, canvas=w)
    else:
        print('No further subdivisions make sense :)')

    myEntryDx.delete(0,END)
    myEntryDx.insert(END,str(dXmm))
    setParameters()

    end= time.clock()
    print('subdiv time :'+str(end - start))

def simplifyArray():
    '''
    This function simplified array - but it take more care to not loose any data
    entered by user
    '''
    global XSecArray, dXmm, dYmm

    if dXmm < 30 and dYmm < 30:
        # Below was working just fine for single phase solver where only 0 or 1 was in the array
        # if np.sum(XSecArray) == 0:
        #     XSecArray = XSecArray[::2,::2] #this is vast and easy but can destory defined Geometry so we do it only for empty array
        # else:
        #     for Row in range(0,XSecArray.shape[0],2):
        #         for Col in range(0,XSecArray.shape[0],2):
        #             # Calculating sume in rows&cols we about to drop
        #             # to be sure we keep all set point transferred
        #
        #             XSecArray[Row,Col] = np.sum(XSecArray[Row:Row+2,Col:Col+2])
        #             if XSecArray[Row,Col] > 0: XSecArray[Row,Col] = 1

        XSecArray = XSecArray[::2,::2]


        dXmm = dXmm*2
        dYmm = dYmm*2

        print(str(dXmm)+'[mm] :'+str(dYmm)+'[mm]')
        csd.n_printTheArray(dataArray=XSecArray, canvas=w)
    else:
        print('No further simplification make sense :)')

    myEntryDx.delete(0,END)
    myEntryDx.insert(END,str(dXmm))
    setParameters()

def showMeForces(*arg):
    '''
    This function abnalyze the cross section array and returns vector of all set
    (equal to 1) elements. This allows to minimize the size of further calculation
    arrays only to active elements.

    and for te moment do the all math for calulations.
    '''
    global elementsVector, resultsArray, resultsCurrentVector, frequency, powerLosses,resultsArrayPower, powerLossesVector


    # Read the setup params from GUI
    setParameters()

    #lets check if there is anything in the xsection geom array
    if np.sum(XSecArray) > 0:
        # We get vectors for each phase`
        elementsVectorPhA = csd.n_arrayVectorize(inputArray=XSecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm)
        elementsVectorPhB = csd.n_arrayVectorize(inputArray=XSecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm)
        elementsVectorPhC = csd.n_arrayVectorize(inputArray=XSecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm)
        # Experimental use of force calculator 
        # I = 1.0*187e3
        # Fa, Fb, Fc = csd.n_getForces(XsecArr=XSecArray,
        #                              vPhA=elementsVectorPhA,
        #                              vPhB=elementsVectorPhB,
        #                              vPhC=elementsVectorPhC,
        #                              Ia=-0.5*I, Ib=-0.5*I, Ic=I)

        # print('Forces: \nA:{}\nB:{}\nC:{}'.format(Fa, Fb, Fc))
        
        root = Tk()
        gui.forceWindow(root,
                        XSecArray,
                        elementsVectorPhA,
                        elementsVectorPhB,
                        elementsVectorPhC)

def vectorizeTheArray(*arg):
    '''
    This function abnalyze the cross section array and returns vector of all set
    (equal to 1) elements. This allows to minimize the size of further calculation
    arrays only to active elements.

    and for te moment do the all math for calulations.
    '''
    global elementsVector, resultsArray, resultsCurrentVector, frequency, powerLosses,resultsArrayPower, powerLossesVector

    # Read the setup params from GUI
    setParameters()

    #lets check if there is anything in the xsection geom array
    if np.sum(XSecArray) > 0:
        # We get vectors for each phase`
        elementsVectorPhA = csd.n_arrayVectorize(inputArray=XSecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm)
        elementsVectorPhB = csd.n_arrayVectorize(inputArray=XSecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm)
        elementsVectorPhC = csd.n_arrayVectorize(inputArray=XSecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm)
        # From here is the rest of calulations
        perymeterA = csd.n_perymiter(elementsVectorPhA, XSecArray, dXmm, dYmm)
        perymeterB = csd.n_perymiter(elementsVectorPhB, XSecArray, dXmm, dYmm)
        perymeterC = csd.n_perymiter(elementsVectorPhC, XSecArray, dXmm, dYmm)

        # memorize the number of elements in each phase
        elementsPhaseA = elementsVectorPhA.shape[0]
        elementsPhaseB = elementsVectorPhB.shape[0]
        elementsPhaseC = elementsVectorPhC.shape[0]

        #Lets put the all phases togethrt
        if elementsPhaseA !=0 and elementsPhaseB != 0 and elementsPhaseC!=0:
            elementsVector = np.concatenate((elementsVectorPhA, elementsVectorPhB, elementsVectorPhC), axis=0)

        elif elementsPhaseA == 0:
            if elementsPhaseB == 0:
                elementsVector = elementsVectorPhC
            elif elementsPhaseC==0:
                elementsVector = elementsVectorPhB
            else:
                elementsVector = np.concatenate((elementsVectorPhB, elementsVectorPhC), axis=0)
        else:
            if elementsPhaseB == 0 and elementsPhaseC == 0:
                elementsVector = elementsVectorPhA
            elif elementsPhaseC == 0:
                elementsVector = np.concatenate((elementsVectorPhA, elementsVectorPhB), axis=0)
            else:
                elementsVector = np.concatenate((elementsVectorPhA, elementsVectorPhC), axis=0)


        print(elementsVector.shape)

        admitanceMatrix = np.linalg.inv(csd.n_getImpedanceArray(csd.n_getDistancesArray(elementsVector),freq=frequency, dXmm=dXmm, dYmm=dYmm, temperature=temperature))

        #Let's put here some voltage vector
        vA = np.ones(elementsPhaseA)
        vB = np.ones(elementsPhaseB)*(-0.5 + (np.sqrt(3)/2)*1j)
        vC = np.ones(elementsPhaseC)*(-0.5 - (np.sqrt(3)/2)*1j)


        voltageVector = np.concatenate((vA,vB,vC), axis=0)

        # Lets calculate the currebt vector as U = ZI >> Z^-1 U = I
        # and Y = Z^-1
        #so finally I = YU - as matrix multiplication goes

        currentVector = np.matmul(admitanceMatrix, voltageVector)

        # And now we need to get solution for each phase to normalize it
        currentPhA = currentVector[0:elementsPhaseA]
        currentPhB = currentVector[elementsPhaseA:elementsPhaseA+elementsPhaseB:1]
        currentPhC = currentVector[elementsPhaseA+elementsPhaseB:]

        # Normalize the solution vectors fr each phase
        currentPhA = currentPhA / csd.n_getComplexModule(np.sum(currentPhA))
        currentPhB = currentPhB / csd.n_getComplexModule(np.sum(currentPhB)) #*(-0.5 + (np.sqrt(3)/2)*1j))
        currentPhC = currentPhC / csd.n_getComplexModule(np.sum(currentPhC)) #*(-0.5 - (np.sqrt(3)/2)*1j))

        # Print out he results currents in each phase
        print('sumy: '+str(csd.n_getComplexModule(np.sum(currentPhA)))+' : '+str(csd.n_getComplexModule(np.sum(currentPhB)))+' : '+str(csd.n_getComplexModule(np.sum(currentPhC)))+' : ')

        print('sumy: '+str((np.sum(currentPhA)))+' : '+str((np.sum(currentPhB)))+' : '+str((np.sum(currentPhC)))+' : ')

        print('Current vector:')
        print(currentVector.shape)
        print('Current vector elements module:')
        getMod = np.vectorize(csd.n_getComplexModule)

        resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)

        resultsCurrentVector = getMod(resultsCurrentVector)
        resistanceVector = csd.n_getResistanceArray(elementsVector, dXmm=dXmm, dYmm=dYmm, temperature=temperature)
        resultsCurrentVector *= curentRMS

        powerLossesVector = resistanceVector * resultsCurrentVector**2
        powerLosses = np.sum(powerLossesVector)

        # Power losses per phase
        powPhA = np.sum(powerLossesVector[0:elementsPhaseA])
        powPhB = np.sum(powerLossesVector[elementsPhaseA:elementsPhaseA+elementsPhaseB:1])
        powPhC = np.sum(powerLossesVector[elementsPhaseA+elementsPhaseB:])
        
        print('power losses: {} [W] \n phA: {}[W]\n phB: {}[W]\n phC: {}[W]'
              .format(powerLosses, powPhA, powPhB, powPhC))

        print('Phases perymeters:\nA: {}mm\nB: {}mm\nC: {}mm\n'
              .format(perymeterA, perymeterB, perymeterC))

        powerLosses = [powerLosses, powPhA, powPhB, powPhC]
        
        # Converting results to form of density
        powerLossesVector /= (dXmm*dYmm)

        # Converting resutls to current density
        resultsCurrentVector /= (dXmm*dYmm)

        # Recreating the solution to form of cross section array
        resultsArray = csd.n_recreateresultsArray(elementsVector=elementsVector, resultsVector=resultsCurrentVector, initialGeometryArray=XSecArray)
        resultsArrayPower = csd.n_recreateresultsArray(elementsVector=elementsVector, resultsVector=powerLossesVector, initialGeometryArray=XSecArray)

        

        #Showing the results
        showResults()

def drawGeometryArray(theArrayToDisplay):

    global figGeom,geomax,geomim

    title_font = { 'size':'11', 'color':'black', 'weight':'normal'}
    axis_font = { 'size':'10'}

    my_cmap = matplotlib.cm.get_cmap('jet')
    my_cmap.set_under('w')

    figGeom = plt.figure(1)

    if np.sum(theArrayToDisplay) == 0:
        vmin=0
    else:
        vmin = 0.8

    geomax = figGeom.add_subplot(1,1,1)

    plotWidth = (theArrayToDisplay.shape[1])*dXmm
    plotHeight = (theArrayToDisplay.shape[0])*dYmm

    geomim = geomax.imshow(theArrayToDisplay, cmap='jet', interpolation='none', extent=[0,plotWidth,plotHeight,0], vmin=vmin)

    geomax.set_xticks(np.arange(0,plotWidth,2*dXmm))
    geomax.set_yticks(np.arange(0,plotHeight,2*dYmm))

    figGeom.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

    plt.xlabel('size [mm]', **axis_font)
    plt.ylabel('size [mm]', **axis_font)
    plt.axis('scaled')
    plt.tight_layout()

    plt.grid(True)
    plt.show()


def showResults():

    title_font = { 'size':'11', 'color':'black', 'weight':'normal'}
    axis_font = { 'size':'10'}

    if np.sum(resultsArray) != 0:

        # Cecking the area in array that is used by geometry to limit the display
        min_row = int(np.min(elementsVector[:,0]))
        max_row = int(np.max(elementsVector[:,0])+1)

        min_col = int(np.min(elementsVector[:,1]))
        max_col = int(np.max(elementsVector[:,1])+1)

        # Cutting down results array to the area with geometry
        resultsArrayDisplay = resultsArray [min_row:max_row,min_col:max_col]
        resultsArrayDisplay2 = resultsArrayPower [min_row:max_row,min_col:max_col]


        # Checking out what are the dimensions od the ploted area
        # to make propper scaling

        plotWidth = (resultsArrayDisplay.shape[1])*dXmm
        plotHeight = (resultsArrayDisplay.shape[0])*dYmm

        fig = plt.figure('Results Window')
        ax = fig.add_subplot(1,1,1)
        
        my_cmap = matplotlib.cm.get_cmap('jet')
        my_cmap.set_under('w')

        im =  ax.imshow(resultsArrayDisplay,   cmap=my_cmap, interpolation='none',  vmin=0.8*np.min(resultsCurrentVector), extent=[0,plotWidth,plotHeight,0])
        fig.colorbar(im, ax=ax, orientation='vertical',label='Current Density [A/mm$^2$]',alpha=0.5, fraction=0.046 )
        plt.axis('scaled')

        ax.set_title(str(frequency)+'[Hz] / '+str(curentRMS)+'[A] / '+str(temperature) +
                     '[$^o$C]\n Power Losses {0[0]:.2f}[W] \n phA: {0[1]:.2f} phB: {0[2]:.2f} phC: {0[3]:.2f}'.format(powerLosses), **title_font)
        
        plt.xlabel('size [mm]', **axis_font)
        plt.ylabel('size [mm]', **axis_font)

        fig.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()
    else:
        print('No results available! Run the analysis first.')


def mainSetup():
    '''
    This function set up (or reset) all the main elements
    '''
    global temperature, canvas_width, canvas_height, elementsInX, elementsInY, dXmm, dYmm, dX, dY, XSecArray, frequency, resultsArray, curentRMS, globalX, globalY, globalZoom

    globalX = 0
    globalY = 0
    globalZoom = 1

    elementsInX = 2*25
    elementsInY = 2*25

    dXmm = 10
    dYmm = 10



    dX = (canvas_width / elementsInX)
    dY = (canvas_height / elementsInY)

    XSecArray = np.zeros(shape=[elementsInY,elementsInX])
    resultsArray = np.zeros(shape=[elementsInY,elementsInX])

    frequency = 50
    curentRMS = 1000
    temperature = 35



def setParameters(*arg):
    global temperature, frequency, AnalysisFreq, curentRMS, dXmm, dYmm, analysisDX, analysisDY

    frequency = float(myEntry.get())
    if frequency == 0:
        frequency = 1e-5

    curentRMS = float(myEntryI.get())
    temperature = float(myEntryT.get())

    dXmm = float(myEntryDx.get())
    dYmm = dXmm

    AnalysisFreq.config(text= 'frequency: '+str(frequency)+'[Hz]\n Current: '+str(curentRMS)+'[A]\n Temperature: '+str(temperature)+'[deg C]')

    analysisDX.config(text='dx:\n '+str(dXmm)+'[mm]')
    analysisDY.config(text='dy:\n '+str(dYmm)+'[mm]')


######## End of functions definition ############



master = Tk()
master.title( "Cross Section Designer" )

img = PhotoImage(file='CSDico.gif')
master.tk.call('wm', 'iconphoto', master._w, img)

canvas_width = 650
canvas_height = 650

mainSetup()



w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.configure(background='white')
w.grid(row=1, column=1, columnspan=5, rowspan=10, sticky=W+E+N+S, padx=1, pady=1)



# opis = Label(text='Cross Section\n Designer\n v0.1', height=15)
# opis.grid(row=8, column=0,)
print_button_clear = Button(master, text='New Geometry', command=clearArrayAndDisplay, height=2, width=16)
print_button_clear.grid(row=1, column=0, padx=5, pady=5)

print_button_load = Button(master, text='Load from File', command=loadArrayFromFile, height=2, width=16)
print_button_load.grid(row=2, column=0, padx=5, pady=5)

print_button_save = Button(master, text='Save to File', command=saveArrayToFile, height=2, width=16)
print_button_save.grid(row=3, column=0, padx=5, pady=5)

emptyOpis = Label(text='', height=3)
emptyOpis.grid(row=5, column=0,)

print_button_slice = Button(master, text='Subdivide', command=subdivideArray, height=2, width=16)
print_button_slice.grid(row=6, column=0 , padx=5, pady=5)

print_button_slice = Button(master, text='Simplify', command=simplifyArray, height=2, width=16)
print_button_slice.grid(row=7, column=0 , padx=5, pady=5)

print_button_zoom = Button(master, text='Zoom In', command=zoomIn, height=2, width=16)
print_button_zoom.grid(row=8, column=0 , padx=5, pady=5)
print_button_zoom = Button(master, text='Zoom Out', command=zoomOut, height=2, width=16)
print_button_zoom.grid(row=9, column=0 , padx=5, pady=5)

print_button_zoom = Button(master, text='<', command=zoomL, height=1, width=1, repeatdelay=100, repeatinterval=100)
print_button_zoom.grid(row=10, column=0 , padx=5, pady=5)
print_button_zoom = Button(master, text='>', command=zoomR, height=1, width=1, repeatdelay=100, repeatinterval=100)
print_button_zoom.grid(row=10, column=1 , padx=5, pady=5)
print_button_zoom = Button(master, text='^', command=zoomU, height=1, width=1, repeatdelay=100, repeatinterval=100)
print_button_zoom.grid(row=11, column=0 , padx=5, pady=5)
print_button_zoom = Button(master, text='v', command=zoomD, height=1, width=1, repeatdelay=100, repeatinterval=100)
print_button_zoom.grid(row=11, column=1 , padx=5, pady=5)

print_button = Button(master, text='Run Analysis!', command=vectorizeTheArray, height=2, width=16)
print_button.grid(row=9, column=8, columnspan=2)


print_button = Button(master, text='Show Results', command=showResults, height=2, width=16)
print_button.grid(row=10, column=8, padx=5, pady=5,columnspan=2)

print_button = Button(master, text='Show Forces', command=showMeForces, height=2, width=16)
print_button.grid(row=11, column=8, padx=5, pady=5,columnspan=2)

GeometryOpis = Label(text='Geometry setup:', height=1)
GeometryOpis.grid(row=0, column=8,columnspan=2)

# AnalysisOpis = Label(text='Analysis setup:', height=3)
# AnalysisOpis.grid(row=4, column=3,columnspan=2)

print_button = Button(master, text='Set parameters', command=setParameters, height=2, width=16)
print_button.grid(row=8, column=8, padx=5, pady=5,columnspan=2)
AnalysisFreq = Label(text= 'frequency: '+str(frequency)+'[Hz]\n Current: '+str(curentRMS)+'[A]\n Temperature: '+str(temperature)+'[deg C]', height=3  )
AnalysisFreq.grid(row=5, column=8,columnspan=2)

myEntry = Entry(master, width = 5 )
myEntry.insert(END,str(frequency))
myEntry.grid(row=6, column=8, padx=1, pady=1)
myEntry.bind("<Return>", setParameters)
myEntry.bind("<FocusOut>", setParameters)

myEntryI = Entry(master, width = 5)
myEntryI.insert(END,str(curentRMS))
myEntryI.grid(row=6, column=9, padx=1, pady=1)
myEntryI.bind("<Return>", setParameters)
myEntryI.bind("<FocusOut>", setParameters)

myEntryT = Entry(master, width = 5 )
myEntryT.insert(END,str(temperature))
myEntryT.grid(row=7, column=8, padx=1, pady=1)
myEntryT.bind("<Return>", setParameters)
myEntryT.bind("<FocusOut>", setParameters)

analysisDX = Label(text='dx\n '+str(dXmm)+'[mm]', height=2  )
analysisDX.grid(row=1, column=8,columnspan=1)
analysisDY = Label(text='dy\n '+str(dYmm)+'[mm]', height=2  )
analysisDY.grid(row=2, column=9,columnspan=1)

wsmall = Canvas(master,width=35,height=35)
wsmall.configure(background='white')
wsmall.grid(row=2, column=8 )

myEntryDx = Entry(master, width = 5)
myEntryDx.insert(END,str(dXmm))
myEntryDx.grid(row=3, column=8, columnspan=2, padx=1, pady=1)
myEntryDx.bind("<Return>", setParameters)
myEntryDx.bind("<FocusOut>", setParameters)



w.bind( "<Button 1>", setPoint)
w.bind( "<Button 3>", resetPoint)
w.bind( "<B1-Motion>", setPoint)
w.bind( "<B3-Motion>", resetPoint)

w.bind( "<Button 2>", showXsecArray)

w.bind("<Left>",zoomL)
w.bind("<Right>",zoomR)
w.bind("<Up>",zoomU)
w.bind("<Down>",zoomD)

message = Label( master, text = "use: Left Mouse Button to Set conductor, Right to reset" )
#message.pack( side = BOTTOM )
message.grid(row=12, column=0, columnspan=3)

phase = IntVar()

phase.set(1) # initialize

Radiobutton(master, text="Phase A", variable=phase, value=1 , indicatoron=0 ,height=1, width=16, bg='red', highlightbackground='red').grid(row=0, column=1)
Radiobutton(master, text="Phase B", variable=phase, value=2 , indicatoron=0 ,height=1, width=16, bg='green', highlightbackground='green').grid(row=0, column=2)
Radiobutton(master, text="Phase C", variable=phase, value=3 , indicatoron=0 ,height=1, width=16, bg='blue', highlightbackground='blue').grid(row=0, column=3)

print_button = Button(master, text='Show / Refresh CAD view', command=displayArrayAsImage, height=1, width=22)
print_button.grid(row=0, column=5, padx=5, pady=0)


master.resizable(width=False, height=False)
master.update()

canvas_height = w.winfo_height()
canvas_width  = w.winfo_width()

csd.n_printTheArray(dataArray=XSecArray, canvas=w)


print(phase)

mainloop()
