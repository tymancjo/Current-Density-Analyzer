# CSD Tool: Analyzing Current Density Distribution in Conductors

## Overview

The CSD (Conductor Section Distribution) Tool is designed to analyze and visualize the current density distribution within conductors of arbitrary cross-sectional shapes. This tool employs a novel approach by discretizing the conductor's cross-section into a fine grid of small, uniform square elements. By treating each element as an individual current-carrying wire, the tool accurately models the complex interactions that govern current flow.

## Core Concept: Discretization and Element Modeling

Instead of analyzing the conductor as a continuous solid, the CSD Tool adopts a discrete representation. The key assumption is:

> The conductor's cross-section is approximated by a grid of small, square elements, where the current density within each element is considered uniform.

This discretization allows us to represent the continuous conductor as a system of discrete "wires." For each of these elemental wires, the tool calculates its fundamental electrical properties:

- **Resistance:** Determined by the element's dimensions and the material conductivity.
- **Self-Inductance:** The inductance of the element due to its own magnetic field.
- **Mutual Inductance:** The inductive coupling between this element and every other element in the discretized system.

## System Modeling: Impedance and Admittance Matrices

By calculating these electrical properties for each element and their interactions, the CSD Tool constructs a comprehensive system of equations. For a single elemental wire, the relationship between voltage and current is expressed as:

$$
U_{wire} = I_{wire} \cdot Z_{wire} + \sum_{n}{I_{wire_{n}}\cdot M_{wire \times wire_{n}}}
$$

Where:
- $U_{wire}$ is the voltage across the elemental wire.
- $I_{wire}$ is the current flowing through the elemental wire.
- $Z_{wire}$ is the self-impedance of the elemental wire (incorporating resistance and self-inductance).
- $I_{wire_{n}}$ is the current flowing through another elemental wire ($n$).
- $M_{wire \times wire_{n}}$ is the mutual inductance between the elemental wire and the other wire ($n$).

This set of equations can be elegantly represented in a more general matrix form for the entire system:

$$
\mathbf{U} = \mathbf{I} \cdot \mathbf{Z}
$$

Where:
- $\mathbf{U}$ is the vector of voltages across all elemental wires.
- $\mathbf{I}$ is the vector of currents flowing through all elemental wires.
- $\mathbf{Z}$ is the impedance matrix, containing the self and mutual impedances of all element pairs.

To solve for the current distribution ($\mathbf{I}$), we can invert the impedance matrix:

$$
\mathbf{I} = \mathbf{U} \cdot \mathbf{Z}^{-1} = \mathbf{U} \cdot \mathbf{G}
$$

Where $\mathbf{G} = \mathbf{Z}^{-1}$ is the admittance matrix.

## Algorithmic Approach: Iterative Current Solution

The typical use case involves defining the total current for each phase of the conductor. However, our core equation uses voltage as the source condition. To bridge this gap, the CSD Tool employs an iterative algorithmic approach:

1.  **Impedance Matrix Calculation:** The tool first calculates the impedance matrix ($\mathbf{Z}$) by systematically analyzing the electrical properties and mutual interactions of each elemental wire with all other wires in the system.
2.  **Admittance Matrix Inversion:** The impedance matrix is then inverted to obtain the admittance matrix ($\mathbf{G}$).
3.  **Initial Voltage Assumption:** An initial voltage vector ($\mathbf{U}$) is assumed for all elements based on the expected phase voltages.
4.  **Current Calculation:** Using the admittance matrix and the assumed voltage vector, the currents in each elemental wire are calculated ($\mathbf{I} = \mathbf{U} \cdot \mathbf{G}$).
5.  **Phase Current Summation and Comparison:** The calculated currents for all elements belonging to the same phase are summed. These summed currents are then compared to the user-defined target currents for each phase.
6.  **Voltage Adjustment:** Based on the difference between the calculated and target phase currents, the source voltage vector ($\mathbf{U}$) is adjusted accordingly.
7.  **Iteration:** Steps 4-6 are repeated iteratively until the calculated phase currents converge to the desired target values however single iteration is normally good enough.

## Output: Current Density Distribution

Through this iterative process, the CSD Tool determines the current flowing through each individual elemental wire. Since the area of each square element is known, we can directly derive the current density (current per unit area) for each part of the conductor's cross-section. This detailed information provides a comprehensive understanding of the current distribution within the conductor.

## Incorporating Magnetic Material Properties

The latest evolution of the tool, newcliCSD, introduces the capability to account for the magnetic properties of materials present within the analyzed geometry during the calculation of self and mutual inductances.

This implementation employs an approximate method that estimates the effective magnetic permeability experienced by each elemental "wire." This estimation is based on a weighted average of the magnetic permeabilities of the surrounding materials, where the weighting factor is inversely proportional to the distance from the elemental wire to each material region.

By integrating this approach, newcliCSD enables the observation and analysis of how paramagnetic materials within the system's geometry can influence the resulting current density distribution. This provides a more realistic and comprehensive simulation of electromagnetic behavior in complex conductor arrangements.