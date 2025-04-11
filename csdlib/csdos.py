""" This library file contains the procedures and functions that support general iterations with the os"""


class Material:
    def __init__(self,name, sigma, alpha, ro, cp,mi_r=1,thermal_conductivity=400):
        self.name = name
        self.sigma = sigma
        self.alpha = alpha
        self.ro = ro
        self.cp = cp
        self.mi_r = mi_r
        self.thermal_conductivity = thermal_conductivity


def read_file_to_list(file_path):
    """
    Reads a text file line by line and stores each line in a list.
    If the file does not exist, an error message is printed.

    :param file_path: Path to the text file.
    :return: List containing each line of the file or an empty list if the file doesn't exist.
    """
    try:
        lines = []
        with open(file_path, "r") as file:
            for line in file:
                lines.append(
                    line.strip()
                )  # .strip() removes any leading/trailing whitespace, including the newline character
        return lines

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []


def get_material_from_list(materials, delim=";", inputs=7):
    material_library = []
    for line in materials:
        material = line.split(delim)
        if len(material) == inputs:
            name = str(material[0])
            sigma = float(material[1])
            alpha = float(material[2])
            ro = float(material[3])
            cp = float(material[4])
            mi_r = float(material[5])
            thermal_conductivity = float(material[6])
            material_library.append(Material(name, sigma, alpha, ro, cp,mi_r,thermal_conductivity))

    return material_library
