import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog, messagebox

import functools
import numpy as np

from multiprocessing import Pool



def checkered(canvas, line_distanceX, line_distanceY):
    '''
    This function clean the board and draw grid
    '''
    # Cleaning up the whole space
    w.create_rectangle(0, 0, canvas_width, canvas_height, fill="white", outline="gray")
    # vertical lines at an interval of "line_distance" pixel
    for x in range(0,canvas_width,int(line_distanceX)):
        canvas.create_line(x, 0, x, canvas_height, fill="gray")
    # horizontal lines at an interval of "line_distance" pixel
    for y in range(0,canvas_height,int(line_distanceY)):
        canvas.create_line(0, y, canvas_width, y, fill="gray")

def arrayVectorize(inputArray,phaseNumber):
    '''
    Desription:
    This function returns vector of 4 dimension vectors that deliver

    input:
    inputArray = 3D array thet describe by 1's position of
    conductors in cross section

    Output:
    [0,1,2,3]

    0 - Oryginal inputArray geometry origin Row for the set cell
    1 - Oryginal inputArray geometry origin Col for the set cell
    2 - X position in mm of the current element
    3 - Y position in mm of the current element

    Number of such [A,B,C,D] elements is equal to the number of defined
    conductor cells in geometry

    '''
    # Let's check the size of the array
    elementsInY = inputArray.shape[0]
    elementsInX = inputArray.shape[1]

    #lets define the empty vectorArray
    vectorArray = []

    #lets go for each input array position and check if is set
    #and if yes then put it into putput vectorArray
    for Row in range(elementsInY):
        for Col in range(elementsInX):
            if inputArray[Row][Col] == phaseNumber:
                # Let's calculate the X and Y coordinates
                coordinateY = (0.5 + Row) * dYmm
                coordinateX = (0.5 + Col) * dXmm

                vectorArray.append([Row,Col,coordinateX,coordinateY])

    return np.array(vectorArray)

def getDistancesArray(inputVector):
    '''
    This function calculate the array of distances between every conductors element
    Input:
    the vector of conductor elements as delivered by vectorizeTheArray
    '''
    # lets check for the numbers of elements
    elements = inputVector.shape[0]
    print(elements)
    # Define the outpur array
    distanceArray = np.zeros((elements, elements))

    for x in range(elements):
        for y in range(elements):
            if x != y:
                posXa =  inputVector[y][2]
                posYa =  inputVector[y][3]

                posXb =  inputVector[x][2]
                posYb =  inputVector[x][3]

                distanceArray[y,x] = np.sqrt((posXa-posXb)**2 + (posYa-posYb)**2)
            else:
                distanceArray[y,x] = 0
    return distanceArray

def getResistance(sizeX, sizeY, lenght, temp, sigma20C, temCoRe):
    '''
    Calculate the resistance of the al'a square shape in given temperature
    All dimensions in mm
    temperature in deg C

    output:
    Resistance in Ohm
    '''
    return (lenght/(sizeX*sizeY*sigma20C)) * 1e3 *(1+temCoRe*(temp-20))


def getSelfInductance(sizeX, sizeY, lenght):
    '''
    Calculate the self inductance for the subconductor
    '''
    srednica = (sizeX+sizeY)/2
    return 0.000000001*2*100*lenght*1e-3*(np.log(2*lenght*1e-3/(0.5*srednica*1e-3))-(3/4))

def getMutualInductance(sizeX, sizeY, lenght, distance):
    '''
    Calculate the mutual inductance for the pait of subconductors
    '''
    srednica = (sizeX+sizeY)/2

    return 0.000000001*2*lenght*1e-1*(np.log(2*lenght*1e-1/(distance/10))-(3/4))

def getImpedanceArray(distanceArray, freq):
    '''
    Calculate the array of impedance as complex values for each element
    Input:
    The elements vector as delivered by arrayVectorize
    freq = frequency in Hz
    '''
    omega = 2*np.pi*freq

    impedanceArray = np.zeros((distanceArray.shape),dtype=np.complex_)
    for X in range(distanceArray.shape[0]):
        for Y in range(distanceArray.shape[0]):
            if X == Y:
                impedanceArray[Y, X] = getResistance(sizeX=dXmm, sizeY=dYmm, lenght=1000, temp=temperature, sigma20C=58e6, temCoRe=3.9e-3) + 1j*omega*getSelfInductance(sizeX=dXmm, sizeY=dYmm, lenght=1000)
            else:
                impedanceArray[Y, X] = 1j*omega*getMutualInductance(sizeX=dXmm, sizeY=dYmm, lenght=1000, distance=distanceArray[Y,X])

    return impedanceArray


def getResistanceArray(elementsVector):
    '''
    Calculate the array of resistance values for each element
    Input:
    The elements vector as delivered by arrayVectorize
    '''

    resistanceArray = np.zeros(elementsVector.shape[0])
    for element in range(elementsVector.shape[0]):

        resistanceArray[element] = getResistance(sizeX=dXmm, sizeY=dYmm, lenght=1000, temp=temperature, sigma20C=58e6, temCoRe=3.9e-3)
    return resistanceArray

def arraySlicer(inputArray, subDivisions):
    '''
    This function increase the resolution of the cross section array
    '''
    return inputArray.repeat(subDivisions,axis=0).repeat(subDivisions,axis=1)

def showXsecArray(event):
    '''
    This function print the array to the terminal
    '''
    print(XSecArray)


def displayArrayAsImage():
    '''
    This function print the array to termianl and shows additional info of the
    dX and dy size in mm
    and redraw the array on the graphical working area
    '''
    print(XSecArray)
    print(str(dXmm)+'[mm] :'+str(dYmm)+'[mm]')
    printTheArray(XSecArray)

    drawGeometryArray(XSecArray)

def clearArrayAndDisplay():
    '''
    This function erase the datat form array and return it back to initial
    setup
    '''
    global XSecArray, dX, dY
    if np.sum(XSecArray) != 0: # Test if there is anything draw on the array
        q = messagebox.askquestion("Delete", "This will delete current shape. Are You Sure?", icon='warning')
        if q == 'yes':
            XSecArray *= 0
            #checkered(w, dX, dY)
            mainSetup()
    else:
            XSecArray *= 0
            #checkered(w, dX, dY)
            mainSetup()
    checkered(w, dX, dY)
    myEntryDx.delete(0,END)
    myEntryDx.insert(END,str(dXmm))
    setParameters()

def saveArrayToFile():
    '''
    This function saves the data of cross section array to file
    '''
    filename = filedialog.asksaveasfilename()
    if filename:
        saveTheData(filename)

def loadArrayFromFile():
    '''
    This function loads the data from the file
    !!!!! Need some work - it dosn't reset properly the dXmm and dYmm
    '''
    if np.sum(XSecArray) != 0: # Test if there is anything draw on the array
        q = messagebox.askquestion("Delete", "This will delete current shape. Are You Sure?", icon='warning')
        if q == 'yes':
            filename = filedialog.askopenfilename()
            if filename:
                loadTheData(filename)
    else:
        filename = filedialog.askopenfilename()
        if filename:
            loadTheData(filename)




def saveTheData(filename):
    '''
    This is the subfunction for saving data
    '''
    print('Saving to file :' + filename)
    np.save(filename, XSecArray)

def loadTheData(filename):
    '''
    This is sub function to load data
    '''
    global XSecArray
    print('Readinf from file :' + filename)
    XSecArray =  np.load(filename)
    printTheArray(XSecArray)




def setUpPoint( event, Set ):
    '''
    This function track the mouse position from event ad setup or reset propper element
    in the cross section array
    '''

    Col = int(event.x/dX)
    Row = int(event.y/dY)

    if event.x < canvas_width and event.y < canvas_height and event.x > 0 and event.y > 0:
        inCanvas = True
    else:
        inCanvas = False


    if Set and inCanvas:
        actualPhase = phase.get()

        if actualPhase == 3:
            w.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="blue", outline="gray")
            XSecArray[Row][Col] = 3
        elif actualPhase == 2:
            w.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="green", outline="gray")
            XSecArray[Row][Col] = 2
        else:
            w.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="red", outline="gray")
            XSecArray[Row][Col] = 1

        try:
            geomim.set_data(XSecArray)
            plt.draw()
        except:
            pass
        # drawGeometryArray(XSecArray)

    elif not(Set) and inCanvas:
        w.create_rectangle(Col*dX, Row*dY, Col*dX+dX, Row*dY+dY, fill="white", outline="gray")
        XSecArray[Row][Col] = 0
        try:
            geomim.set_data(XSecArray)
            plt.draw()
        except:
            pass

        # drawGeometryArray(XSecArray)

def printTheArray(dataArray):
    '''
    This function allows to print the array back to the graphical board
    usefull for redraw or draw loaded data
    '''
    global dX, dY
    # Let's check the size
    elementsInY = dataArray.shape[0]
    elementsInX = dataArray.shape[1]

    # Now we calculate the propper dX and dY for this array
    dX = (canvas_width / (elementsInX))
    dY = (canvas_height / (elementsInY))

    # Now we cleanUp the field
    checkered(w, dX, dY)

    for Row in range(elementsInY):
        for Col in range(elementsInX):
            if dataArray[Row][Col] == 1:
                fillColor = "red"
            elif dataArray[Row][Col] == 2:
                fillColor = "green"
            elif dataArray[Row][Col] == 3:
                fillColor = "blue"
            else:
                fillColor = "white"

            w.create_rectangle((Col)*dX, (Row)*dY, (Col)*dX+dX, (Row)*dY+dY, fill=fillColor, outline="gray")

def subdivideArray():
    '''
    This function is logical wrapper for array slicer
    it take care to not loose any entered data from the modufied array
    '''
    global XSecArray, dXmm, dYmm
    if dXmm > 1 and dYmm > 1:
        XSecArray = arraySlicer(inputArray = XSecArray, subDivisions = 2)

        dXmm = dXmm/2
        dYmm = dYmm/2
        print(str(dXmm)+'[mm] :'+str(dYmm)+'[mm]')
        printTheArray(dataArray=XSecArray)
    else:
        print('No further subdivisions make sense :)')

    myEntryDx.delete(0,END)
    myEntryDx.insert(END,str(dXmm))
    setParameters()


def simplifyArray():
    '''
    This function simplified array - but it take more care to not loose any data
    entered by user
    '''
    global XSecArray, dXmm, dYmm

    if dXmm < 15 and dYmm < 15:

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
        printTheArray(dataArray=XSecArray)
    else:
        print('No further simplification make sense :)')

    myEntryDx.delete(0,END)
    myEntryDx.insert(END,str(dXmm))
    setParameters()


def recreateresultsArray(elementsVector, resultsVector, initialGeometryArray):

    localResultsArray = np.zeros((initialGeometryArray.shape), dtype=float)

    vectorIndex = 0
    for result in resultsVector:
        localResultsArray[int(elementsVector[vectorIndex][0]),int(elementsVector[vectorIndex][1])] = result
        vectorIndex +=1

    return localResultsArray


def getComplexModule(x):
    '''
    returns the module of complex number
    input: x - complex number
    '''
    return np.sqrt(x.real**2 + x.imag**2)


def runMainAnalysisHT():
    '''
    Experimental function for HT calculation
    '''
    p=Pool(4)
    p.map(vectorizeTheArray)

def vectorizeTheArray(*arg):
    '''
    This function abnalyze the cross section array and returns vector of all set
    (equal to 1) elements. This allows to minimize the size of further calculation
    arrays only to active elements.

    and for te moment do the all math for calulations.
    '''
    global elementsVector, resultsArray, resultsCurrentVector, frequency, powerLosses

    # Read the setup params from GUI
    setParameters()

    #lets check if there is anything in the xsection geom array
    if np.sum(XSecArray) > 0:
        # We get vectors for each phase`
        elementsVectorPhA = arrayVectorize(inputArray=XSecArray, phaseNumber=1)
        elementsVectorPhB = arrayVectorize(inputArray=XSecArray, phaseNumber=2)
        elementsVectorPhC = arrayVectorize(inputArray=XSecArray, phaseNumber=3)
        # From here is the rest of calulations

        #memorize the number of elements in each phase
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
        # print(elementsVector)
        # print(getDistancesArray(elementsVector))
        admitanceMatrix = np.linalg.inv(getImpedanceArray(getDistancesArray(elementsVector),freq=frequency))
        # print('Calculated addmintance Matrix:')
        # print(admitanceMatrix)

        #Let's put here some voltage vector
        vA = np.ones(elementsPhaseA)
        vB = np.ones(elementsPhaseB)*(-0.5 + (np.sqrt(3)/2)*1j)
        vC = np.ones(elementsPhaseC)*(-0.5 - (np.sqrt(3)/2)*1j)


        voltageVector = np.concatenate((vA,vB,vC), axis=0)

        print('Voltage vector:')
        print(voltageVector.shape)

        # Lets calculate the currebt vector as U = ZI >> Z^-1 U = I
        # and Y = Z^-1
        #so finally I = YU - as matrix multiplication goes

        currentVector = np.matmul(admitanceMatrix, voltageVector)
        print('pierwotnie obliczony I')
        print(currentVector)
        # And now we need to get solution for each phase to normalize it
        currentPhA = currentVector[0:elementsPhaseA]
        currentPhB = currentVector[elementsPhaseA:elementsPhaseA+elementsPhaseB:1]
        currentPhC = currentVector[elementsPhaseA+elementsPhaseB:]

        print('wektory rozdzielone')
        print(currentPhA)
        print(currentPhB)
        print(currentPhC)


        currentPhA = currentPhA / getComplexModule(np.sum(currentPhA))
        currentPhB = currentPhB / getComplexModule(np.sum(currentPhB)) #*(-0.5 + (np.sqrt(3)/2)*1j))
        currentPhC = currentPhC / getComplexModule(np.sum(currentPhC)) #*(-0.5 - (np.sqrt(3)/2)*1j))

        print('sumy: '+str(getComplexModule(np.sum(currentPhA)))+' : '+str(getComplexModule(np.sum(currentPhB)))+' : '+str(getComplexModule(np.sum(currentPhC)))+' : ')

        print('sumy: '+str((np.sum(currentPhA)))+' : '+str((np.sum(currentPhB)))+' : '+str((np.sum(currentPhC)))+' : ')

        print('Current vector:')
        print(currentVector.shape)
        print('Current vector elements module:')
        getMod = np.vectorize(getComplexModule)

        resultsCurrentVector = np.concatenate((currentPhA,currentPhB,currentPhC), axis=0)

        # print(getMod(resultsCurrentVector))
        # print(np.sum(resultsCurrentVector))
        # print(getComplexModule(np.sum(resultsCurrentVector)))

        resultsCurrentVector = getMod(resultsCurrentVector)
        resistanceVector = getResistanceArray(elementsVector)
        resultsCurrentVector *= curentRMS

        # print('vector currents shape')
        # print(resultsCurrentVector.shape)
        # print('Resistance shape')
        # print(resistanceVector.shape)

        powerLossesVector = resistanceVector * resultsCurrentVector**2

        powerLosses = np.sum(powerLossesVector)

        resultsCurrentVector /= (dXmm*dYmm)
        print(powerLosses)

        resultsArray = recreateresultsArray(elementsVector=elementsVector, resultsVector=resultsCurrentVector, initialGeometryArray=XSecArray)

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
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        my_cmap = matplotlib.cm.get_cmap('jet')
        my_cmap.set_under('w')

        # Cecking the area in array that is used by geometry to limit the display
        min_row = int(np.min(elementsVector[:,0]))
        max_row = int(np.max(elementsVector[:,0])+1)

        min_col = int(np.min(elementsVector[:,1]))
        max_col = int(np.max(elementsVector[:,1])+1)

        # Cutting down results array to the area with geometry
        resultsArrayDisplay = resultsArray [min_row:max_row,min_col:max_col]

        # Checking out what are the dimensions od the ploted area
        # to make propper scaling

        plotWidth = (resultsArrayDisplay.shape[1]+1)*dXmm
        plotHeight = (resultsArrayDisplay.shape[0]+1)*dYmm

        im = ax.imshow(resultsArrayDisplay, cmap='jet', interpolation='none',  vmin=0.9*np.min(resultsCurrentVector), extent=[0,plotWidth,plotHeight,0])

        if plotWidth < plotHeight:
            fig.colorbar(im, orientation='vertical',label='Current Density [A/mm2]',alpha=0.5)
        else:
            fig.colorbar(im, orientation='horizontal',label='Current Density [A/mm2]',alpha=0.5)

        plt.axis('scaled')

        ax.set_title(str(frequency)+'[Hz] / '+str(curentRMS)+'[A] / '+str(temperature)+'[$^o$C]\n Power losses: '+str(round(powerLosses,2))+'[W]', **title_font)

        plt.xlabel('size [mm]', **axis_font)
        plt.ylabel('size [mm]', **axis_font)

        plt.tight_layout()
        plt.show()
    else:
        print('No results available! Run the analysis first.')


def mainSetup():
    '''
    This function set up (or reset) all the main elements
    '''
    global temperature, canvas_width, canvas_height, elementsInX, elementsInY, dXmm, dYmm, dX, dY, XSecArray, frequency, resultsArray, curentRMS


    elementsInX = 2*25
    elementsInY = 2*25

    dXmm = 10
    dYmm = 10


    dX = int(canvas_width / elementsInX)
    dY = int(canvas_height / elementsInY)

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

canvas_width = 750
canvas_height = 750

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


print_button = Button(master, text='Run Analysis!', command=vectorizeTheArray, height=2, width=16)
print_button.grid(row=9, column=8, columnspan=2)


print_button = Button(master, text='Show Results', command=showResults, height=2, width=16)
print_button.grid(row=10, column=8, padx=5, pady=5,columnspan=2)

GeometryOpis = Label(text='Geometry setup:', height=3)
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



w.bind( "<Button 1>", functools.partial(setUpPoint, Set=True))
w.bind( "<Button 3>", functools.partial(setUpPoint, Set=False))
w.bind( "<B1-Motion>", functools.partial(setUpPoint, Set=True))
w.bind( "<B3-Motion>", functools.partial(setUpPoint, Set=False))

w.bind( "<Button 2>", showXsecArray)


message = Label( master, text = "use: Left Mouse Button to Set conductor, Right to reset" )
#message.pack( side = BOTTOM )
message.grid(row=11, column=0, columnspan=3)

phase = IntVar()

phase.set(1) # initialize

Radiobutton(master, text="Phase A", variable=phase, value=1 , indicatoron=0 ,height=2, width=16, bg='red', highlightbackground='red').grid(row=0, column=1)
Radiobutton(master, text="Phase B", variable=phase, value=2 , indicatoron=0 ,height=2, width=16, bg='green', highlightbackground='green').grid(row=0, column=2)
Radiobutton(master, text="Phase C", variable=phase, value=3 , indicatoron=0 ,height=2, width=16, bg='blue', highlightbackground='blue').grid(row=0, column=3)

print_button = Button(master, text='Show / Refresh CAD view', command=displayArrayAsImage, height=2, width=22)
print_button.grid(row=0, column=5, padx=5, pady=0)


master.resizable(width=False, height=False)


checkered(w, dX, dY)

print(phase)

mainloop()
