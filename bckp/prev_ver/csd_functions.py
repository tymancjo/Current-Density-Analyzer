import os

def myLog(s: str = "", *args, **kwargs):
    if verbose:
        print(s, *args, *kwargs)

def loadTheData(filename):
    """
    This is sub function to load data
    """

    if os.path.isfile(filename):
        _, extension = os.path.splitext(filename)
        myLog("File type: " + extension)

        if extension.lower() in [".txt", ".inc", ".ic"]:
            myLog("reading the inner-code geometry file: " + filename)
            try:
                with open(filename, "r") as f:
                    file_content = f.read()

                XSecArray, dXmm, dYmm = getCanvas(file_content)
            except IOError:
                print("Error reading the file " + filename)
                sys.exit(1)

        else:
            myLog("reading from file :" + filename)
            XSecArray, dXmm, dYmm = loadObj(filename).restore()

        return XSecArray, dXmm, dYmm
    else:
        myLog(f"The file {filename} can't be opened!")
        sys.exit(1)
