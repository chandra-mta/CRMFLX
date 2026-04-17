import numpy as np
from importlib.resources import files, as_file
from . import _model_data

MAXKP = 9 #: NUMBER OF KP BINS IN THE DATA FILE
MAXNUM = 65 #: MAXIMUM NUMBER OF VOLUME ELEMENTS ALONG AN AXIS USED IN DATAFILE

_RESOURCES = files(_model_data)

__all__ = ["load_model"]

def _parse_data_file(model_name):
    """read input ascii file and initialize the numpy datarrays

    :param model_name: Name of the model for parsing the data file
    :type model_name: str
    """
    traversable = _RESOURCES.joinpath(f"{model_name.lower()}.asc")
    with as_file(traversable) as _path:
        with open(_path) as f:
            data = [line.strip() for line in f.readlines()]
    return data

def load_model(model, calc_imapindx=False):
    """
    output: xflux   --- array of arrays of containing the x-coordinate of each data cell's center (re)
            yflux   --- array of arrays of containing the y-coordinate of each data cell's center (re)
            zflux   --- array of arrays of containing the z-coordinate of each data cell's center (re)
            flxbin  --- array of arrays of the average ion flux within each cell  (ions/[cm^2-sec-sr-mev])
            numbin  --- array of arrays of the number of non-zero values within each cell
            Note: sub arrays have different length and you cannot apply normal numpy operation
            numdat  --- number of non-zero values in the database
            imapindx--- array of pointers to flux database
    """
    data = _parse_data_file(model)
    xflux  = []
    yflux  = []
    zflux  = []
    flxbin = []
    numbin = []
    numdat = []
    #: the data lists are created for each kp values between 1 and 9 (0 - 8 columns)
    for k in range(0, MAXKP):
        xflux.append([])
        yflux.append([])
        zflux.append([])
        flxbin.append([])
        numbin.append([])
        numdat.append(0)
    
    if calc_imapindx:
        imapindx = np.full((MAXKP, MAXNUM, MAXNUM, MAXNUM), -9999, int)

    for ent in data:
        #: Data file indexes from 1.
        atemp = ent.split()
        xidx = int(atemp[0]) -1
        yidx = int(atemp[1]) -1
        zidx = int(atemp[2]) -1
        xpos = float(atemp[3])
        ypos = float(atemp[4])
        zpos = float(atemp[5])
#
#--- in the table, flxbin data are from column 6 to 14 and numbin1 data are from 15 to 23 
#--- (column start from 0) maxkp columns each
#
        for k in range(0, MAXKP):
            fval = float(atemp[6+k])
            nval = int(atemp[15+k])

            if nval > 0:
                #: Fewer data points for higher KPs
                xflux[k].append(xpos)
                yflux[k].append(ypos)
                zflux[k].append(zpos)
                flxbin[k].append(fval)
                numbin[k].append(nval)

                if calc_imapindx:
                    imapindx[k][xidx][yidx][zidx] = int(numdat[k])
                numdat[k] += 1
#
#--- since the lists in the lists are all different length, we need to handle 
#--- separately to convert into an array
#
    #: Use object dtype so arrays of different lengths become an array-of-arrays
    xflux   = np.array([np.array(xi) for xi in xflux], dtype=object)
    yflux   = np.array([np.array(xi) for xi in yflux], dtype=object)
    zflux   = np.array([np.array(xi) for xi in zflux], dtype=object)
    flxbin  = np.array([np.array(xi) for xi in flxbin], dtype=object)
    numbin  = np.array([np.array(xi).astype(int) for xi in numbin], dtype=object)
    numdat  = np.array(numdat, dtype=int)

    _dict = {
            'xflux': xflux,
            'yflux': yflux,
            'zflux': zflux,
            'flxbin': flxbin,
            'numbin': numbin,
            'numdat': numdat
            }
    if calc_imapindx:
        _dict['imapindx'] = imapindx
    return _dict