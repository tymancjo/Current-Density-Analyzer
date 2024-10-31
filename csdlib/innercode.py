"""This is the library set for the inner code operations"""

### General geometry generators ###

def addCircle(x0,y0,D1,Set, D2=0,Set2=0,draw=True,shift=(0,0),XSecArray=None,dXmm=1):
    """Generalized formula to add circle at given position (x,y) [mm]
    of a two diameters external D1 and internal D2 (if a donat is needed) [mm]"""

    if draw:
        # this works on global canvas array
        # global XSecArray

        x0 = x0 - shift[0]
        y0 = y0 - shift[1]
        
        r1sq = (D1/2)**2
        r2sq = (D2/2)**2

        elementsInY = XSecArray.shape[0]
        elementsInX = XSecArray.shape[1]
            
        for x in range(elementsInX):
            for y in range(elementsInY):
                xmm = x*dXmm+dXmm/2
                ymm = y*dXmm+dXmm/2
                distSq = (xmm - x0)**2 + (ymm-y0)**2 
                if  distSq < r2sq :
                    XSecArray[y,x] = Set2
                elif distSq <= r1sq:
                    XSecArray[y,x] = Set

    x0 = x0 - D1/2
    y0 = y0 - D1/2
    xE = x0 + D1
    yE = y0 + D1
    return [x0,y0,xE,yE]

    
def addRect(x0,y0,W,H,Set,draw=True, shift=(0,0),XSecArray=None,dXmm=1):
    """Generalized formula to add rectangle at given position 
    start - left top corner(x,y)[mm]
    width, height[mm]"""

    xE = x0 + W
    yE = y0 + H

    if draw:
        # this works on global canvas array
        # global XSecArray
        x0 = x0 - shift[0]
        y0 = y0 - shift[1]
        xE = x0 + W
        yE = y0 + H
        

        elementsInY = XSecArray.shape[0]
        elementsInX = XSecArray.shape[1]
            
        for x in range(elementsInX):
            for y in range(elementsInY):
                xmm = x*dXmm+dXmm/2
                ymm = y*dXmm+dXmm/2

                if (x0 <= xmm <= xE) and (y0 <= ymm <= yE) :
                    XSecArray[y,x] = Set

    return [x0,y0,xE,yE]



def codeLoops(input_text):

    commands = {
        'l': [None,[1]]
    }

    for line_nr,line in enumerate(reversed(input_text)):
        index_non_rev = len(input_text) - 1 - line_nr
        if len(line)>3:
            command = line[0].lower()
            if command in commands:
                if command == 'l':
                    # taking care of looping the stuff
                    ar = line[2:-1].split(',')
                    if len(ar) in commands[command][1]:
                        loops = int(ar[0]) - 1 # the -1 is due to the text is aready there once. 
                        loop_code = input_text[index_non_rev+1:]
                        # cleaning this loop code line
                        input_text[index_non_rev] = "\n"
                        for _ in range(loops):
                            input_text.extend(loop_code)
                        codeLoops(input_text)

    return input_text

def textToCode(input_text):
    """This is the function that will return the list 
    of geometry execution code stps.
    Code commands are in the form of dictionary"""

    commands = {
        'c': [addCircle,[4,6]],
        'r': [addRect,[5]],
        'v': [None,[2]],
        'a': [None, [2]],
        'l': [None,[1]]
    }

    innerCodeSteps = []
    innerVariables = {}

    # loops are separate function as they need to be recursion for nested loops
    input_text = codeLoops(input_text)

    for line_nr,line in enumerate(input_text):
        if len(line)>5:
            command = line[0].lower()
            if command in commands:
                if command == 'v':
                    # taking care if the command sets the variable
                    ar = line[2:-1].strip().split(',')
                    if len(ar) in commands[command][1]:
                        variable_name = str(ar[0])
                        variable_value = float(ar[1])
                        innerVariables[variable_name] = variable_value
                elif command == 'a':
                    if len(innerVariables):
                        # taking care if the command sets the variable
                        ar = line[2:-1].strip().split(',')
                        if len(ar) in commands[command][1]:
                            variable_name = str(ar[0])
                            variable_value = innerVariables[variable_name]+float(ar[1])
                            innerVariables[variable_name] = variable_value

                else:
                    # ar as argumnents 
                    ar = line[2:-1].strip().split(',')
                    # insert inner variables if any
                    if len(innerVariables):
                        # let's replace the variables with values
                        for i,argument in enumerate(ar):
                            if argument in innerVariables:
                                ar[i] = innerVariables[argument]

                    if len(ar) in commands[command][1]:
                        ar = [float(a) for a in ar]
                        innerCodeSteps.append([commands[command][0],ar,command])

    return innerCodeSteps




               