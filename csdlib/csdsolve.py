""""
The module with the solver related functions. 
"""
import sys
import numpy as np
from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm


def solve_system(
    XsecArray,
    dXmm,
    dYmm,
    I,
    freq,
    length,
    temperature,
    verbose=False,
    sigma20C=56e6,
    temCoRe=3.9e-3,
):
    # lets workout the  current in phases as is defined
    in_Ia = I[0] * np.cos(I[1] * np.pi / 180) + I[0] * np.sin(I[1] * np.pi / 180) * 1j
    csdf.myLog(f"Ia: {in_Ia}")

    in_Ib = I[2] * np.cos(I[3] * np.pi / 180) + I[2] * np.sin(I[3] * np.pi / 180) * 1j
    csdf.myLog(f"Ib: {in_Ib}")

    in_Ic = I[4] * np.cos(I[5] * np.pi / 180) + I[4] * np.sin(I[5] * np.pi / 180) * 1j
    csdf.myLog(f"Ic: {in_Ic}")

    vPhA = csdm.arrayVectorize(
        inputArray=XsecArray, phaseNumber=1, dXmm=dXmm, dYmm=dYmm
    )
    vPhB = csdm.arrayVectorize(
        inputArray=XsecArray, phaseNumber=2, dXmm=dXmm, dYmm=dYmm
    )
    vPhC = csdm.arrayVectorize(
        inputArray=XsecArray, phaseNumber=3, dXmm=dXmm, dYmm=dYmm
    )

    (
        elementsVector,
        elementsPhaseA,
        elementsPhaseB,
        elementsPhaseC,
    ) = csdf.combineVectors(vPhA, vPhB, vPhC)

    if len(elementsVector) > 1200:
        csdf.myLog()
        csdf.myLog(
            "!!! Size of the elements vector may lead to very long calculation. Be aware!"
        )
        csdf.myLog("You can break the process by CTRL+C")
        csdf.myLog("You may conceder reduce the split steps.")
        csdf.myLog("Optimal element size is around 1.5x1.5mm")
        csdf.myLog()

    if len(elementsVector) > 10000:
        csdf.myLog("Extreme size of elements vector - long calculations immanent!")

    admitanceMatrix = csdm.getGmatrix(
        csdm.getImpedanceArray(
            csdm.getDistancesArray(elementsVector),
            freq=freq,
            dXmm=dXmm,
            dYmm=dYmm,
            temperature=temperature,
            lenght=length,
            sigma20C=sigma20C,
            temCoRe=temCoRe,
        )
    )

    # Let's put here some voltage vector
    # Ua = complex(1, 0)
    # Ub = complex(-0.5, np.sqrt(3) / 2)
    # Uc = complex(-0.5, -np.sqrt(3) / 2)

    Ua = np.cos(I[1] * np.pi / 180) + np.sin(I[1] * np.pi / 180) * 1j
    Ub = np.cos(I[3] * np.pi / 180) + np.sin(I[3] * np.pi / 180) * 1j
    Uc = np.cos(I[5] * np.pi / 180) + np.sin(I[5] * np.pi / 180) * 1j

    vA = np.ones(elementsPhaseA) * Ua
    vB = np.ones(elementsPhaseB) * Ub
    vC = np.ones(elementsPhaseC) * Uc

    # voltageVector = np.concatenate((vA, vB, vC), axis=0)
    voltageVector, _, _, _ = csdf.combineVectors(vA, vB, vC)
    # print(f"{voltageVector=}, {len(voltageVector)=}")
    currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringing each phase current to the assumed Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)

    # expected Ia Ib Ic as symmetrical ones
    # ratios of currents will give us new voltages for phases

    if Ia:
        Ua = Ua * (in_Ia / Ia)
    if Ib:
        Ub = Ub * (in_Ib / Ib)
    if Ic:
        Uc = Uc * (in_Ic / Ic)

    csdf.myLog()
    csdf.myLog("Calculated require Source Voltages")
    csdf.myLog(Ua)
    csdf.myLog(Ub)
    csdf.myLog(Uc)

    # Setting up the voltage vector for final solve
    vA = np.ones(elementsPhaseA) * Ua
    vB = np.ones(elementsPhaseB) * Ub
    vC = np.ones(elementsPhaseC) * Uc

    voltageVector = np.concatenate((vA, vB, vC), axis=0)

    # Final solve
    # Main equation solve
    currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)

    currentPhA = currentVector[0:elementsPhaseA]
    currentPhB = currentVector[elementsPhaseA : elementsPhaseA + elementsPhaseB]
    currentPhC = currentVector[elementsPhaseA + elementsPhaseB :]

    # Bringing each phase current to the assumer Irms level
    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)
    # end of second solve!

    csdf.myLog()
    csdf.myLog("Solution check...")
    csdf.myLog("Raw Current results:")
    csdf.myLog(f"Ia: {Ia}")
    csdf.myLog(f"Ib: {Ib}")
    csdf.myLog(f"Ic: {Ic}")
    csdf.myLog()
    csdf.myLog(f"Sum: {Ia+Ib+Ic}")

    # Now we normalize up to the expecter I - just a polish
    # as we are almost there with the previous second solve for new VOLTAGES
    modIa = np.abs(Ia)
    modIb = np.abs(Ib)
    modIc = np.abs(Ic)

    if modIa:
        currentPhA *= in_Ia / modIa
    if modIb:
        currentPhB *= in_Ib / modIb
    if modIc:
        currentPhC *= in_Ic / modIc

    Ia = np.sum(currentPhA)
    Ib = np.sum(currentPhB)
    Ic = np.sum(currentPhC)

    csdf.myLog("Fix Current results:")
    csdf.myLog(f"Ia: {Ia}")
    csdf.myLog(f"Ib: {Ib}")
    csdf.myLog(f"Ic: {Ic}")
    csdf.myLog()
    csdf.myLog(f"Sum: {Ia+Ib+Ic}")

    # Data postprocessing
    getMod = np.vectorize(csdm.getComplexModule)

    resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)
    resultsCurrentVector = getMod(resultsCurrentVector)

    resistanceVector = csdm.getResistanceArray(
        elementsVector,
        dXmm=dXmm,
        dYmm=dYmm,
        temperature=temperature,
        lenght=length,
        sigma20C=sigma20C,
        temCoRe=temCoRe,
    )

    # This is the total power losses vector
    powerLossesVector = resistanceVector * resultsCurrentVector**2
    # This are the total power losses
    powerLosses = np.sum(powerLossesVector)

    # Power losses per phase
    powPhA = np.sum(powerLossesVector[0:elementsPhaseA])
    powPhB = np.sum(
        powerLossesVector[elementsPhaseA : elementsPhaseA + elementsPhaseB : 1]
    )
    powPhC = np.sum(powerLossesVector[elementsPhaseA + elementsPhaseB :])

    return (
        resultsCurrentVector,
        (powerLosses, powPhA, powPhB, powPhC),
        elementsVector,
        powerLossesVector,
    )



def solve_multi_system(
    XsecArray,
    dXmm,
    dYmm,
    I,
    freq,
    length,
    temperature,
    verbose=False,
    sigma20C=56e6,
    temCoRe=3.9e-3,
):

    # Determining the number of phases
    array_of_phases = np.unique(XsecArray)
    number_of_phases = len(np.unique(XsecArray))-1 #as there is an 0 for empty in this

    # Let's check if the delivered currents data are given for all phases
    currents = []
    if len(I) == number_of_phases:
        for i in I:
            this_current = i[0] * np.cos(i[1] * np.pi / 180) + i[0] * np.sin(i[1] * np.pi / 180) * 1j
            currents.append(this_current)
    else:
        csdf.myLog(f"Error! Found {number_of_phases=} and {len(I)} currents definitions. Mismatch!")
        sys.exit(1)
        
    csdf.myLog(f"Currents {currents=}")


    # Let's work out the phases elements vectors
    vPh = []
    elementsPhase = []
    for p in array_of_phases:
        if p != 0:
            this_vPh = csdm.arrayVectorize(
                        inputArray=XsecArray, phaseNumber=int(p), dXmm=dXmm, dYmm=dYmm)
            vPh.append(this_vPh)
            elementsPhase.append(len(this_vPh))

    elementsVector = np.concatenate(vPh)

    csdf.myLog(f"{elementsVector=}")
    csdf.myLog(f"{elementsPhase=}")

    if len(elementsVector) > 1200:
        csdf.myLog()
        csdf.myLog(
            "!!! Size of the elements vector may lead to very long calculation. Be aware!"
        )
        csdf.myLog("You can break the process by CTRL+C")
        csdf.myLog("You may conceder reduce the split steps.")
        csdf.myLog("Optimal element size is around 1.5x1.5mm")
        csdf.myLog()

    if len(elementsVector) > 10000:
        csdf.myLog("Extreme size of elements vector - long calculations immanent!")

    admitanceMatrix = csdm.getGmatrix(
        csdm.getImpedanceArray(
            csdm.getDistancesArray(elementsVector),
            freq=freq,
            dXmm=dXmm,
            dYmm=dYmm,
            temperature=temperature,
            lenght=length,
            sigma20C=sigma20C,
            temCoRe=temCoRe,
        )
    )

    # Let's put here some voltage vector

    U = []
    for i in I:
        this_u = np.cos(i[1] * np.pi / 180) + np.sin(i[1] * np.pi / 180) * 1j
        U.append(this_u)
    
    voltageVector = np.array([])
    for elements, u in zip(elementsPhase, U):
        this_v = np.ones(elements) * u
        voltageVector = np.concatenate((voltageVector,this_v), axis=0)

    currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)

    # And now we need to get solution for each phase to normalize it
    currentsPh = []
    start_index = 0
    for l in elementsPhase:
        end_index = start_index + l
        currentsPh.append(currentVector[start_index:end_index])
        start_index = end_index
    
    # Bringing each phase current to the assumed Irms level
    I_results = [np.sum(cPh) for cPh in currentsPh]

    # Normalizing to get the currents as in input parameters
    # ratios of currents will give us new voltages for phases

    csdf.myLog(f"Initial Voltages {U=}")
    for n, x  in enumerate(zip(I_results, currents, U)):
        i_r, i,u = x
        if i_r:
            this_Z = u / i_r
            U[n] = this_Z * i
    csdf.myLog(f"Modified Voltages {U=}")
    

    # Setting up the voltage vector for final solve
    voltageVector = np.array([])
    for elements, u in zip(elementsPhase, U):
        this_v = np.ones(elements) * u
        voltageVector = np.concatenate((voltageVector,this_v), axis=0)

    # Final solve
    # Main equation solve
    currentVector = csdm.solveTheEquation(admitanceMatrix, voltageVector)

    currentsPh = []
    start_index = 0
    for l in elementsPhase:
        end_index = start_index + l
        currentsPh.append(currentVector[start_index:end_index])
        start_index = end_index

    # Bringing each phase current to the assumed Irms level
    I_results = [np.sum(cPh) for cPh in currentsPh]
    # end of second solve!

    csdf.myLog()
    csdf.myLog("Solution check...")
    csdf.myLog("Raw Current results:")
    csdf.myLog(f"I: {I_results=}")
    csdf.myLog()
    csdf.myLog(f"Sum: {np.sum(I_results):.3f}")

    # Now we normalize up to the expecter I - just a polish
    # as we are almost there with the previous second solve for new VOLTAGES
    modI = [np.abs(cPh) for cPh in I_results]
    csdf.myLog(f"Mod: {modI=}")

    for n,x in enumerate(zip(modI,currentsPh,I)):
        mod_i,cPh,i = x
        if mod_i != 0 and i[0]!=0:
            currentsPh[n] = cPh * i[0]/mod_i

    I_results = [np.sum(cPh) for cPh in currentsPh]

    csdf.myLog("Fix Current results:")
    csdf.myLog(f"I: {I_results=}")
    csdf.myLog()
    csdf.myLog(f"Sum: {np.sum(I_results):.3f}")
    modI = [np.abs(cPh) for cPh in I_results]
    csdf.myLog(f"Mod: {modI=}")

    currentVector = np.concatenate(currentsPh)

    # Data postprocessing
    getMod = np.vectorize(csdm.getComplexModule)

    # resultsCurrentVector = np.concatenate((currentPhA, currentPhB, currentPhC), axis=0)
    resultsCurrentVector = getMod(currentVector)

    resistanceVector = csdm.getResistanceArray(
        elementsVector,
        dXmm=dXmm,
        dYmm=dYmm,
        temperature=temperature,
        lenght=length,
        sigma20C=sigma20C,
        temCoRe=temCoRe,
    )

    # This is the total power losses vector
    powerLossesVector = resistanceVector * resultsCurrentVector**2
    # This are the total power losses
    powerLosses = np.sum(powerLossesVector)

    powPh = []
    start_index = 0
    for l in elementsPhase:
        end_index = start_index + l
        powPh.append(np.sum(powerLossesVector[start_index:end_index]))
        start_index = end_index


    return (
        resultsCurrentVector,
        (powerLosses, powPh),
        elementsVector,
        powerLossesVector,
        currentVector,
        vPh,
    )

