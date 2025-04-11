"""
The module with the solver related functions.
"""

import sys
import numpy as np
from csdlib import csdfunctions as csdf
from csdlib import csdmath as csdm
from csdlib import csdos


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
    phases_material=[],
    sigma20C=56e6,
    temCoRe=3.9e-3,
):

    # Figuring out phases material
    if len(phases_material) == 0:
        phases_material.append(csdos.Material("Cu", sigma, temCoRe))

    # Determining the number of phases
    list_of_phases = np.unique(XsecArray).astype(int)
    number_of_phases = (
        len(np.unique(XsecArray)) - 1
    )  # as there is an 0 for empty in this
    phase_index = {phase: index for index, phase in enumerate(list_of_phases)}

    # Let's check if the delivered currents data are given for all phases
    currents = []
    if len(I) == number_of_phases:
        for i in I:
            this_current = (
                i[0] * np.cos(i[1] * np.pi / 180)
                + i[0] * np.sin(i[1] * np.pi / 180) * 1j
            )
            currents.append(this_current)
    else:
        csdf.myLog(
            f"Error! Found {number_of_phases=} and {len(I)} currents definitions. Mismatch!"
        )
        sys.exit(1)

    if len(phases_material) != number_of_phases:
        temp = phases_material[0]
        phases_material = []
        for _ in range(number_of_phases):
            phases_material.append(temp)

    csdf.myLog(f"Currents {currents=}")
    csdf.myLog(f"MAterials {phases_material}")

    # Let's work out the phases elements vectors
    vPh = []
    elementsPhase = []
    for p in list_of_phases:
        if p != 0:
            this_vPh = csdm.arrayVectorize(
                inputArray=XsecArray, phaseNumber=int(p), dXmm=dXmm, dYmm=dYmm
            )
            vPh.append(this_vPh)
            elementsPhase.append(len(this_vPh))

    elementsVector = np.concatenate(vPh)

    # we need to prepare the vector versions of the material properties and temperatures
    sigma_array = []
    alpha_array = []
    for element in elementsVector:
        index = phase_index[int(element[4])] - 1
        sigma_array.append(phases_material[index].sigma)
        alpha_array.append(phases_material[index].alpha)

    sigma_array = np.array(sigma_array)
    alpha_array = np.array(alpha_array)

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
            sigma20C=sigma_array,
            temCoRe=alpha_array,
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
        voltageVector = np.concatenate((voltageVector, this_v), axis=0)

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
    for n, x in enumerate(zip(I_results, currents, U)):
        i_r, i, u = x
        if i_r:
            this_Z = u / i_r
            U[n] = this_Z * i
    csdf.myLog(f"Modified Voltages {U=}")

    # Setting up the voltage vector for final solve
    voltageVector = np.array([])
    for elements, u in zip(elementsPhase, U):
        this_v = np.ones(elements) * u
        voltageVector = np.concatenate((voltageVector, this_v), axis=0)

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

    for n, x in enumerate(zip(modI, currentsPh, I)):
        mod_i, cPh, i = x
        if mod_i != 0 and i[0] != 0:
            currentsPh[n] = cPh * i[0] / mod_i

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
        sigma_array=sigma_array,
        alpha_array=alpha_array,
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


def solve_with_magnetic(
    XsecArr: np.array,
    phases_materials,
    dXmm,
    dYmm,
    currents,
    frequency,
    length,
    temperature,
    verbose=False,
):
    """
    This is the new version of the solver intended to incorporate the magnetic
    influence of the sketched geometry. The idea is to have solver that uses the
    magnetic permeability of the materials and the analysis domain.
    This shall lead to calculate impact on the inductances L and M
    and by those to observe the impact of presence of the paramagnetic materials like
    carbon steel and by this the impact on current distribution.


    The plan:
    - no need to be 100% compatible with previous versions
    1. Get the geometry data, materials data, current data, dXmm and dYmm data.
    2. Create the m_r array - reflecting the geometry data
    3. Create the m_r_weighted array - having calculated the weighted average for each coordinates
    4. Prepare the Admittance Array using the 2 and 3 when calculating L and M
    5. Solve for currents - like in previous solvers - with the same normalization approach

    Inputs:
    XsecArray - the 2D np array with the geometry
    phases_materials - the dictionary with phase materials definitions
    dXmm, dYmm - real world size of the cell in [mm]
    currents - the phase currents dictionary.
    frequency - in [Hz]
    length - the length of the analysis [mm]
    temperature - the temperature of the analyzed conductors in [deg C]
    """

    # figuring out the list of phases in geometry
    list_of_phases = np.unique(XsecArr).astype(int)
    # removing the 0 as it's an empty space not phase:
    if 0 in list_of_phases:
        list_of_phases = list_of_phases[list_of_phases != 0]

    number_of_phases = len(list_of_phases)

    csdf.myLog(f"normalized: {list_of_phases=}")
    phase_index_dict = {phase: index for index, phase in enumerate(list_of_phases)}

    # checking for the data consistency
    # and preparing the starting data
    if len(currents) != number_of_phases:
        print(f"Error! Currents definition don't match phases!")
        csdf.myLog(f"{currents=}")
        csdf.myLog(f"{list_of_phases=}")
        sys.exit(1)
    else:
        I = []
        for i in currents:
            Imod = float(i[0])
            Phi = float(i[1]) * np.pi / 180
            I.append(Imod * (np.cos(Phi) + 1j * np.sin(Phi)))

    if len(phases_materials) != number_of_phases:
        print(f"Error! Material definition don't match phases!")
        sys.exit(1)

    # preparing the material properties geometry like arrays
    # how to make it in the best way? Many arrays or one with
    # more dimensions?

    # Let's go for the higher dimension ones:
    # the structure: [sigma, alpha, m_r material, m_r weighted]
    idx_sigma = 0
    idx_alpha = 1
    idx_m_r = 2
    idx_m_r_w = 3

    Rows, Cols = XsecArr.shape

    # making mi_r array for the geometry
    mi_r_array = np.copy(XsecArr).astype(int)
    # taking care of the empty space
    mi_r_array[mi_r_array == 0] = 1

    for phase in list_of_phases:
        phase_index = phase_index_dict[phase]
        mi_r = phases_materials[phase_index].mi_r
        mi_r_array[mi_r_array == phase] = mi_r

    csdf.myLog("Starting with mi_r weighted...")

    mi_r_weighted_array = csdm.get_mi_weighted(XsecArr, mi_r_array, dXmm, delta=250)
    # mi_r_weighted_array = csdm.get_mi_averaged(XsecArr, mi_r_array, dXmm, delta=10) 

    csdf.myLog("...")

    # gathering all material and domain data to one matrix
    materials_Xsec_array = np.ones((Rows, Cols, 4))
    for R in range(Rows):
        for C in range(Cols):
            phase = int(XsecArr[R, C])
            if phase != 0:
                phase_index = phase_index_dict[phase]
                material = phases_materials[phase_index]

                materials_Xsec_array[R, C, idx_sigma] = material.sigma
                materials_Xsec_array[R, C, idx_alpha] = material.alpha
                materials_Xsec_array[R, C, idx_m_r] = mi_r_array[R, C]
                materials_Xsec_array[R, C, idx_m_r_w] = mi_r_weighted_array[R, C]

    # Let's work out the phases elements vectors
    vPh = []
    elementsPhase = []
    for p in list_of_phases:
        if p != 0:
            this_vPh = csdm.arrayVectorize(
                inputArray=XsecArr, phaseNumber=int(p), dXmm=dXmm, dYmm=dYmm
            )
            vPh.append(this_vPh)
            elementsPhase.append(len(this_vPh))

    elementsVector = np.concatenate(vPh)
    sigma_array = []
    alpha_array = []
    mi_r_array = []
    mi_r_w_array = []

    for element in elementsVector:
        r = int(element[0])
        c = int(element[1])
        sigma_array.append(materials_Xsec_array[r, c, idx_sigma])
        alpha_array.append(materials_Xsec_array[r, c, idx_alpha])
        mi_r_array.append(materials_Xsec_array[r, c, idx_m_r])
        mi_r_w_array.append(materials_Xsec_array[r, c, idx_m_r_w])

    sigma_array = np.array(sigma_array)
    alpha_array = np.array(alpha_array)
    mi_r_array = np.array(mi_r_array)
    mi_r_w_array = np.array(mi_r_w_array)

    csdf.myLog("Finished Pre Process...")
    csdf.myLog(f"Prepared {I=}")

    # preparing for the first solve
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

    # getting the admittance matrix with all the materials used
    admitanceMatrix = csdm.getGmatrix(
        csdm.getImpedanceArray(
            csdm.getDistancesArray(elementsVector),
            freq=frequency,
            dXmm=dXmm,
            dYmm=dYmm,
            temperature=temperature,
            lenght=length,
            sigma20C=sigma_array,
            temCoRe=alpha_array,
            mi_r=mi_r_array,
            mi_r_w=mi_r_w_array,
            use_mi_array=True,
        )
    )

    # rest of solve is as in the previous version
    #############################################

    # Let's put here some voltage vector
    U = []
    for i in currents:
        Imod = float(i[0])
        Phi = float(i[1]) * np.pi / 180
        U.append(np.cos(Phi) + np.sin(Phi) * 1j)

    voltageVector = np.array([])
    for elements, u in zip(elementsPhase, U):
        this_v = np.ones(elements) * u
        voltageVector = np.concatenate((voltageVector, this_v), axis=0)

    # SOLVE #
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
    for n, x in enumerate(zip(I_results, I, U)):
        i_r, i, u = x
        if i_r:
            this_Z = u / i_r
            U[n] = this_Z * i
    csdf.myLog(f"Modified Voltages {U=}")

    # Setting up the voltage vector for final solve
    voltageVector = np.array([])
    for elements, u in zip(elementsPhase, U):
        this_v = np.ones(elements) * u
        voltageVector = np.concatenate((voltageVector, this_v), axis=0)

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

    for n, x in enumerate(zip(modI, currentsPh, currents)):
        mod_i, cPh, i = x
        if mod_i != 0 and i[0] != 0:
            currentsPh[n] = cPh * i[0] / mod_i

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
        sigma_array=sigma_array,
        alpha_array=alpha_array,
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
        mi_r_weighted_array
    )

def solve_thermal_for_bars(list_of_bars, HTC=5):
    ...
    # we need to make the thermal conductivity array for the whole system.
    # this will be a np array of B X B size, where B is the number of bars. 
    B = len(list_of_bars)


    thermal_G_matrix_cond = np.zeros((B,B),dtype = float)
    thermal_G_matrix = np.zeros((B,B),dtype = float)
    vector_Q = np.zeros((B),dtype=float)

    ## For the first approach we combine the phases together at the ends (both ends) thermally 
    for r,bar_n in enumerate(list_of_bars):
        for c,bar_m in enumerate(list_of_bars):
            if bar_n.phase == bar_m.phase and bar_n is not bar_m:
                thermal_G_matrix_cond[r,c] = 2*(1 / (bar_n.Rth+bar_m.Rth))

    for r,bar_n in enumerate(list_of_bars):
        for c,bar_m in enumerate(list_of_bars):
            if bar_n is bar_m:
                thermal_G_matrix[r,c] =  bar_n.length * bar_n.perymiter*1e-6 * HTC + thermal_G_matrix_cond[r,:].sum()
                vector_Q[r] = bar_n.power
            else:
                thermal_G_matrix[r,c] = -thermal_G_matrix_cond[r,c]
    
    inverse_G_matrix = np.linalg.inv(thermal_G_matrix)
    dT_vector = np.matmul(inverse_G_matrix,vector_Q)

    for bar,dt in zip(list_of_bars,dT_vector):
        bar.dT = dt

    


    
