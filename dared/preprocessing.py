from glob import glob
from astropy.io import fits
import numpy as np

def create_masterfile(directory_list, method=np.median, output="", datatype=np.float64, **kwargs) -> np.ndarray:
    """A utility function that takes a LIST of directories that are looked up for respective fits files. All of the files are taken and according to the averaging method to be used, taken the mean/median of. The method needs to be a function that provides an "axis" option like the np.mean or np.median function. They can be used alternatively as well.
    - You can/have to specify a mask to filter for the fits files only with the "mask" argument
    - "method" specifies the stack operation (default is "numpy.median"). It must be a function, able to operate on a 3-dimensional numpy ndarray, collapsing it on the first of all three axes
    """
    data        = []
    frames      = []
    succ, counter = 0, 0
    for directory in directory_list:
        if "mask" in kwargs:
            frames += glob(directory + kwargs["mask"])
        else:
            if directory[-1] in ["/", "\\"]: directory += "*"
            frames += glob(directory)

    for frame in frames:
        try:
            data.append(fits.getdata(frame))
            succ += 1
        except:
            Warning("Cannot read the data in file:\n\t", frame, "\nMake sure that this is a valid fits file")
        counter += 1
    
    data = np.array(data, dtype=datatype)
    master_data = method(data, axis=0)

    if output != "":
        fits.writeto(filename=output, data=master_data, overwrite=True)
    return master_data