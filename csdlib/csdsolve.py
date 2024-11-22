""""
The module with the solver related functions. 
"""
import numpy as np
from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm


def solve_system(XsecArray, dXmm, dYmm, I, freq, length, temperature, verbose=False):
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
        elementsVector, dXmm=dXmm, dYmm=dYmm, temperature=temperature, lenght=length
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
    )
