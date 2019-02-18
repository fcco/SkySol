import os
import sys
import numpy as np
from datetime import datetime
import pytz
import time
import calendar
import h5py
from skysol.lib import misc
from pandas import Series
from sklearn.externals import joblib

def iniRadData(ini, locations=[], locfile=None):
    """ Initialise pyranometer station instances """

    nstations = 0
    pyr = []
    st = time.time()

    if ini.radiation:

        # Read radiation data from pyranometer stations
        pyr.append(pyranometer(ini))
        j = nstations
        if ini.live:
            pyr[j].read_live(ini.nrt_rad)
        else:
            pyr[j].read_niwe(ini.data_file)
        pyr[-1].ind = 0
        pyr[j].type = "thermopile"
        pyr[j].unit = 'W/m2'
        pyr[-1].mflag = True
        pyr[-1].latitude = ini.lat0
        pyr[-1].longitude = ini.lon0
        pyr[-1].dist2Camera = 0
        brng = 0
        if ini.cbh_flag:
            pyr[-1].map_y, pyr[-1].map_x = misc.polar2grid( pyr[-1].dist2Camera, \
                brng, ini.grid_size, ini.x_res, ini.y_res )

        nstations += 1

    else:
        # add empty station at position of camera
        pyr.append(pyranometer(ini))
        pyr[-1].latitude = ini.lat0
        pyr[-1].longitude = ini.lon0
        pyr[-1].dist2Camera = 0
        brng = 0
        if ini.cbh_flag:
            pyr[-1].map_y, pyr[-1].map_x = misc.polar2grid( pyr[-1].dist2Camera, \
                brng, ini.grid_size, ini.x_res, ini.y_res )
        pyr[-1].ind = 0
        nstations += 1

    # add additional forecast locations
    for i in range(0,len(locations)):
        pyr.append(pyranometer(ini))
        pyr[-1].ind = i+1
        pyr[-1].mflag = False
        pyr[-1].latitude = locations[i][0]
        pyr[-1].longitude = locations[i][1]
        if ini.cbh_flag:
            pyr[-1].dist2Camera = misc.coordDist(ini.lat0,ini.lon0,pyr[-1].latitude,pyr[-1].longitude)
            brng = misc.bearing(ini.lat0,ini.lon0,pyr[-1].latitude,pyr[-1].longitude)
            pyr[-1].map_y, pyr[-1].map_x = misc.polar2grid( pyr[-1].dist2Camera, \
                brng, ini.grid_size, ini.x_res, ini.y_res )
        nstations += 1

    print('Read station data...\tfinished in %.1f seconds' % (time.time() - st))

    return nstations,pyr


# Function for smoothing (e.g. timeseries moving averages)
def smooth(x, window_len=10, window='hanning'):

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')

    return y[window_len-1:-window_len+1]


# Pyranometer Measurements
class pyranometer:

    def __init__(self,ini):

        self.time = []
        self.dates = []
        self.ghi = []
        self.dhi = []
        self.dni = []
        self.pv = []
        self.qflag = []
        self.mflag = False

        # Position
        self.latitude = 0.0
        self.longitude = 0.0

        # Increasing Number
        self.number = -1

        # Index if available
        self.ind = -1

        # Coordinates in Contour Map
        self.map_x = 0
        self.map_y = 0

        # Unit
        self.unit = ""

        # Type of Measurement
        self.type = ""

        self.start_time = 0
        self.end_time = 0

        # Distance to Camera Position
        self.dist2Camera = 0.0

        # Distance to Image Center ( with relation to shadows )
        self.dist2ImCenter = 0.0

        # Actual time index
        self.tind = -1

        # Forecast data
        self.reset_fcst_arrs(ini)

        # analyses of global horizontal
        self.img_len = int(ini.plot_last_vals / ini.camera_res)
        self.aghi = np.empty(self.img_len,dtype=np.float)
        self.aghi[:] = np.nan
        self.aghi_time = np.empty(self.img_len)
        self.aghi_time[:] = np.nan


        # Modeled radiation from image features
        self.dni_csi_cam = [] # direct normal clear sky index
        self.dni_cam = [] # direct normal estimated from image
        self.dni_time = [] # timestamps
        self.dhi_csi_cam = [] # diffuse horizontal clear sky index
        self.dhi_cam = [] # diffuse horizontal estimated from image
        self.dhi_time = [] # timestamps

        self.model = np.empty(int(ini.plot_last_vals / ini.camera_res),dtype=np.float)
        self.model[:] = np.nan

        # additional data for PV modeling
        self.fwind = [] # predicted wind speed
        self.ftem = [] # predicted ambient air temperature

        self.csi_min = 0.4
        self.csi_max = 1.0

    def reset_fcst_arrs(self, ini):

        # Forecast data

        # position ( x,y )
        self.fpos = np.empty([ini.fcst_horizon,2],dtype=np.int16); self.fpos[:,:] = np.nan
        # binary value from cloud mask
        self.bin = np.empty(ini.fcst_horizon,dtype=np.float32); self.fpos[:,:] = np.nan
        # global horizontal
        self.fghi = np.empty(ini.fcst_horizon,dtype=np.float32); self.fghi[:] = np.nan
        # diffuse horizontal
        self.fdhi = np.empty(ini.fcst_horizon,dtype=np.float32); self.fdhi[:] = np.nan
        # direct normal
        self.fdni = np.empty(ini.fcst_horizon,dtype=np.float32); self.fdni[:] = np.nan
        # time
        self.ftime = np.empty(ini.fcst_horizon,dtype=np.float32); self.ftime[:] = np.nan

    def read_live(self,filename):

        import h5pyd, requests.packages.urllib3

        # Disable certificate checking for HTTPS requests for now.
        requests.packages.urllib3.disable_warnings()
        found = False
        cnt = 0
        while cnt < 10:
            try:
                name = filename.split('@')[1]
                ep = filename.split('@')[0]
                os.environ['https_proxy'] = ""
                df = h5pyd.File(name,'r', endpoint=ep)
                sectime = df['Datetime'][:]
                found = True
                break
            except:
                # wait a second
                print(datetime.now(), ' Could not read %s - Try again..' % filename)
                time.sleep(1)
                cnt = cnt + 1
                continue

        if not found:
            print('Cannot read ' + filename + '! -> Exit')
            sys.exit(2)

        self.ghi = df['GHI'][:]
        self.dni = df['DNI'][:]
        self.dhi = df['DHI'][:]

        self.qflag = np.repeat(1,len(self.dhi))
        self.time = np.array(sectime,dtype=np.int64)

        del sectime


    def read(self,filename):

        try:
            f = h5py.File(filename,'r')
        except IOError:
            sys.exit("\n Measurement data: Unable to open file %s" % filename)

        self.ghi = f['GHI'][:]
        #self.dhi = f['DHI'][:]
        #self.dni = f['DNI'][:]
        try:
            self.wspd = f['WindSpd'][:]
            self.wdir = f['WindDir'][:]
            self.tamb = f['Tem'][:]
            self.tmod = f['PV_T1'][:]
            self.poa = f['GHI_Tilt'][:]
        except KeyError:
            pass
        self.time = np.array(f['Datetime'][:],dtype=np.int64)
        self.qflag = np.zeros_like(self.ghi)


        self.ghi = np.array(self.ghi)
        self.time = np.array(self.time)
        self.dni = np.array(self.dni)
        self.dhi = np.array(self.dhi)
        self.qflag = np.array(self.qflag)

        self.dates = [ datetime.utcfromtimestamp(ts) for ts in self.time ]

        del f

    def dtconv_niwe(self,instr):
        instr = instr.decode('UTF-8')
        try:
            dt = datetime.strptime(instr,'"%d-%m-%Y %H:%M:%S"')
            ldt = self.tzi.localize(dt,is_dst=False).astimezone(pytz.UTC)
            return ldt.timestamp()
        except:
            return np.nan



    def read_niwe(self,filename):
        import pandas as pd
        self.tzi = pytz.timezone("Asia/Colombo")
        
        # Read CSV line by line an read the last column
        with open(filename) as f:
            lis=[line.split(',') for line in f]        # create a list of lists
        ts = [l[0] for l in lis[1:]]

        # Convert strings with timestamps to python internal datetime objects
        dts = [ datetime.strptime(d, "%d-%m-%Y %H:%M:%S") for d in ts ]
        # Convert local datetime to UTC datetime to make it compatible with image timestamps
        ldt = [ self.tzi.localize(dt,is_dst=False).astimezone(pytz.UTC) for dt in dts ]

        # Measured GHI
        ghi = [ int(l[-2]) for l in lis[1:]]
        # Unix Timestamp
        time = [ d.timestamp() for d in ldt ]
        # Datetime objects
        dates = np.array([ pytz.UTC.localize(datetime.utcfromtimestamp(ts)) for ts in time ])
        
        # interpolate measurements to 1s resolution
        ghi = Series(ghi, index = dates)
        res = ghi.resample('1S').interpolate()
        self.dates = [ts.to_pydatetime() for ts in res.index]
        self.time = [ dt.timestamp() for dt in self.dates ]
        self.ghi = np.array(res.values)
        self.qflag = np.repeat(0,len(self.ghi))




def getCSI(ini,nstat,pyr,csk,sza,feat,mode="hist"):

    csi = 0.0

    for i in range(0,nstat):

        # set defaults
        pyr[i].csi_min = 0.4; pyr[i].csi_max = 1.0; pyr[i].csi = np.nan; pyr[i].csi_sigma = np.nan
        if not pyr[i].mflag: continue

        ind = pyr[i].tind
        if ind == -1:
            st = -ini.avg_csi; et = None
        else:
            st = ind - ini.avg_csi; et = ind

        # CSI Average last x seconds
        pyr[i].csi = np.nanmean(pyr[i].ghi[st:et] / csk.ghi[csk.tind-ini.avg_csi:csk.tind])
        pyr[i].csi_sigma = np.nanstd(pyr[i].ghi[st:et] / csk.ghi[csk.tind-ini.avg_csi:csk.tind])

        if csi > 0:
            csi = np.average((csi,pyr[i].csi))
        else:
            csi = pyr[i].csi

        if mode == "fix":

            pyr[i].csi_min = 0.4; pyr[i].csi_max = 1.0

        if mode == "hist":
            """
            Histogram method

            The method takes past GHI measurements (how many is user defined)
            and computes the histogram of its clear sky index.

            For mixed conditions two peaks for clear sky and overcast conditions should
            be detectable. Both are taken as lower and upper irradiance limits.
            """

            # CSI clear sky/overcast
            y = np.divide( pyr[i].ghi[ind-ini.avg_csiminmax:ind], csk.ghi[csk.tind-ini.avg_csiminmax:csk.tind] )
            y = y[np.isfinite(y)]
            h, edges = np.histogram(y,bins=ini.hist_bins, density=True,range=(0.2,1.5))
            if np.all(np.isnan(h)):
                continue

            kmin = 0.0; kmax = 0.0
            for k in range(0,len(h)):
                # CSI Overcast ( last 0.5 hour )
                if edges[k] < 0.5 and h[k] > kmin: kmin = h[k]; pyr[i].csi_min = edges[k]
                # CSI Clear Sky ( last 0.5 hour )
                if edges[k] > 0.9 and h[k] > kmax: kmax = h[k]; pyr[i].csi_max = edges[k]

        elif mode == "meas":
            """ The measurement method takes near real time DHI measurements to compute the lower
            irradiance level. The upper level is estimated by adding clear sky DNI

            The method only works with DHI measurements available
            """
            if len(pyr[i].dhi > 0):
                pyr[i].csi_min = np.nanmean(np.divide(pyr[i].dhi[st:et],csk.ghi[csk.tind-ini.avg_csi:csk.tind]))
                directhor = csk.dni[csk.tind] * np.cos(sza)
                pyr[i].csi_max = pyr[0].csi_min + ( directhor / csk.actval )
            else:
                print('No DHI measurements available. Fix values for irradiance are used instead. Choose "hist" method if only GHI is available')
                pyr[i].csi_min = 0.4; pyr[i].csi_max = 1.0

    pyr[0].csi



def dni_clear(pyr, csk, nimg=90, min_data=6):
    """
    Estimate DNI as the difference of measured GHI and estimated DHI. Use previous
    *nimg* images and list only if more than *min_data* values are present
    """
    dni_est = None

    # get indizes
    mask_meas = np.in1d( pyr.time, pyr.dni_time[-nimg:])

    if np.sum(mask_meas) > 0:
        mask_csk = np.in1d( csk.time,pyr.time[mask_meas] )
        mask_time = np.in1d( pyr.dni_time[-nimg:], np.array(csk.time)[mask_csk])
        # get values
        ghi_obs = pyr.ghi[mask_meas]
        dni_clear = csk.dni[mask_csk]
        dhi_est = np.array(pyr.dhi_cam[-nimg:])[mask_time]

        if len(dhi_est) > min_data:
            try:
                dni_est = ( ghi_obs - dhi_est ) / csk.cossza[mask_csk] / dni_clear
            except ValueError:
                dni_est = None

    return dni_est






def dni_level(dni, percentile=0.9, percentage_valid=0.1, lower_limit=0.2):
    """ Determines the maximum DNI from DNI-measurements or DNI-estimations
    given as input.

    This approach assumes that the average of the upper 10 percentiles of the given data
    can be seen as the maximum possible DNI in the given time ( in clear sky,
    under thin cirrus or translucident As,Cs )

    A lower limit for the measured clear sky index can be set.
    The values below are not considered to be act as input for the statistics
    as they correspond to reductions due to thick opaque clouds.

    If the time period is overcast another parameter (percentage_valid)
    determines if under consideration of the lower limit enough values are available for the statistics.
    """
    if dni is None: return -1

    if np.sum(dni > lower_limit) > ( percentage_valid * len(dni) ):
        # only values with dni greater than given limit
        vals = dni[dni > lower_limit]
        # sort in ascending order
        vals.sort()
        # index
        m = int(percentile * len(vals))
        # average of upper percentile
        dnilevel = np.nanmean( vals[m:] )
    else:
        dnilevel = -1

    return dnilevel






def bin2rad(value,csk_value,csi_min=0.4,csi_max=1.0):
    # scale "value" into csi minimum and csi maximum
    if type(value) == np.ndarray:
        value = value.astype(np.float32)
        value[value==0] = np.nan
    val = csi_min * csk_value + (csi_max - csi_min) * csk_value \
         * ( value - 1.0  ) / (255.)
    del value

    return val


def sub_series(pyr,secs):

    ghi = pyr.ghi[pyr.tind-secs:pyr.tind]
    dhi = pyr.dhi[pyr.tind-secs:pyr.tind]
    dni = pyr.dni[pyr.tind-secs:pyr.tind]

    return ghi, dhi, dni
