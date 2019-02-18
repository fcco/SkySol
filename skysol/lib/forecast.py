import numpy as np
from skysol.lib.radiation import bin2rad

def stations(z, nstations, pyr, ini, cskval, csza, tstep):
    """
    applies a mapping from binary cloud map to radiation values according to lower
    and upper boundaries provided for each station as clear sky index
    """

    # Loop through all stations
    for i in range(0, nstations):

        # Get grid cell coordinate
        x = int(pyr[i].fpos[tstep][0])
        y = int(pyr[i].fpos[tstep][1])

        # outside map boundaries?
        if x < 0 or y < 0 or x >= z.shape[0] or y >= z.shape[1] or z[x,y] == 0:
            pyr[i].fghi[tstep] = np.nan
            pyr[i].ftime[tstep] = np.nan
            pyr[i].bin[tstep] = np.nan
            pyr[i].fdhi[tstep] = np.nan
            pyr[i].fdni[tstep] = np.nan
            continue

        # Convert binary value to irradiance value according to clear sky index levels
        # provided by csi_min and csi_max
        val = bin2rad(z[x,y], cskval, csi_min=pyr[i].csi_min, csi_max=pyr[i].csi_max)

        # Binary value from cloud mask
        pyr[i].bin[tstep] = z[x,y]
        # GHI
        pyr[i].fghi[tstep] = val
        # DHI
        pyr[i].fdhi[tstep] = pyr[i].csi_min * cskval
        # DNI
        pyr[i].fdni[tstep] = (pyr[i].fghi[tstep] - pyr[i].fdhi[tstep]) / csza
        # Time information
        pyr[i].ftime[tstep] = tstep
