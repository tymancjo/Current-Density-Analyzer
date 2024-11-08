




def combineVectors(vPhA, vPhB, vPhC):
    """Function is joining the 3 phase vectors together"""

    # Lets put the all phases together
    elementsPhaseA = len(vPhA)
    elementsPhaseB = len(vPhB)
    elementsPhaseC = len(vPhC)

    if elementsPhaseA != 0 and elementsPhaseB != 0 and elementsPhaseC != 0:
        elementsVector = np.concatenate((vPhA, vPhB, vPhC), axis=0)
    elif elementsPhaseA == 0:
        if elementsPhaseB == 0:
            elementsVector = vPhC
        elif elementsPhaseC == 0:
            elementsVector = vPhB
        else:
            elementsVector = np.concatenate((vPhB, vPhC), axis=0)
    else:
        if elementsPhaseB == 0 and elementsPhaseC == 0:
            elementsVector = vPhA
        elif elementsPhaseC == 0:
            elementsVector = np.concatenate((vPhA, vPhB), axis=0)
        else:
            elementsVector = np.concatenate((vPhA, vPhC), axis=0)
    
    return elementsVector, elementsPhaseA,elementsPhaseB, elementsPhaseC