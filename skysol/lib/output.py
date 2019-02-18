from datetime import datetime
import time
import h5py
import os
import numpy as np

def hdfresize(dset,add=1):
    """
    Resize hdf5 dataset
    """

    shape = list(dset.shape)
    shape[0] = shape[0] + add
    dset.resize(shape)

    return dset


def write2Table(filename,init=False,**kwargs):
    """
    Write meta data into table
    """
    h = [ ('%s' % s) for s in sorted(kwargs.keys()) ]
    if "CloudClassProb" in kwargs.keys():
        if type(kwargs['CloudClassProb']) == np.ndarray:
            kwargs['CloudClassProb'] = kwargs['CloudClassProb'][kwargs['CloudClass']-1]
        else:
            kwargs['CloudClassProb'] = np.nan
    l = [ ('%g' % kwargs[t]) if t is not "time" else ('%d' % kwargs[t] ) for s,t in enumerate(sorted(kwargs))]
    if init:
        with open(filename,'w') as fp:
            print(*h, file=fp, sep=" ")
            print(*l, file=fp, sep=" ")
    else:
        with open(filename,'a') as fp:
            print(*l, file=fp, sep=" ")



def write_cmv(filename, times, cmv, append=False):
    """
    Write initial and terminate points of cloud motion vectors in HDF5-file
    """
    if len(cmv.theta) == 0: return
    if append and os.path.exists(filename):
        h = h5py.File(filename,'a')
        dset = h['time']
        dset = hdfresize(dset,add=1)
        dset[-1] = times
    else:
        h = h5py.File(filename,'w')
        time = h.create_dataset('time',shape=(1,), maxshape=(None,),dtype="int64")
        time[0] = times
        time.attrs['Description'] = 'Unix Timestamp'
        time.attrs['Timezone'] = 'UTC'

    grp = h.create_group(str(times))
    dset = grp.create_dataset('theta',shape=cmv.theta.shape,dtype="float32")
    dset[:] = cmv.theta[:]
    dset = grp.create_dataset('azimuth', shape=cmv.azimuth.shape, dtype="float32")
    dset[:] = cmv.azimuth[:]

    h.close()


def write2HDF5(filename, tstamp, ini, init=False,**data):

    ######################
    # Initialise datasets
    ######################

    if init:

        f = h5py.File(filename,'w')

        horizon = int(ini.fcst_horizon / ini.fcst_res)
        time = f.create_dataset('time',(0,),maxshape=(None,),dtype="int64", compression="gzip")
        time.attrs['Description'] = 'Unix Timestamp'
        time.attrs['Timezone'] = 'UTC'

        if ini.fcst_flag:
            fcst = f.create_group('forecast')
            nst = len(data['fcst'])
            fcst.attrs['Temporal resolution in [s]'] = ini.fcst_res
            dset = fcst.create_dataset("fghi", (0,horizon,nst),maxshape=(None,horizon,nst), dtype=np.float32, compression="gzip")
            dset.attrs['Description'] = "Forecast of global horizontal irradiance in W/m^2 along the forecast path"
            dset = fcst.create_dataset("fdhi", (0,horizon,nst),maxshape=(None,horizon,nst), dtype=np.float32, compression="gzip")
            dset.attrs['Description'] = "Forecast of diffuse horizontal irradiance in W/m^2 along the forecast path"
            dset = fcst.create_dataset("fdni", (0,horizon,nst),maxshape=(None,horizon,nst), dtype=np.float32, compression="gzip")
            dset.attrs['Description'] = "Forecast of direct normal irradiance in W/m^2 along the forecast path"
            if not ini.live:
                dset = fcst.create_dataset("mghi", (0,horizon,nst),maxshape=(None,horizon,nst), dtype=np.float32, compression="gzip")
                dset.attrs['Description'] = "Corresponding measurement of global horizontal irradiance in W/m^2 along the forecast path"
            dset = fcst.create_dataset("cghi", (0,horizon),maxshape=(None,horizon), dtype=np.float32, compression="gzip")
            dset.attrs['Description'] = "Corresponding clear sky value of global horizontal irradiance in W/m^2 along the forecast path"

        cmv = f.create_group('cmv')
        dset = cmv.create_dataset("cmv_u", (0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Global CMV u-component used for global forecast vector"
        dset = cmv.create_dataset("cmv_v", (0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Global CMV v-component used for global forecast vector"
        dset = cmv.create_dataset("cmv_spd_sigma", (0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Standard deviation CMV speed of all (good) CMV derived from current image"
        dset = cmv.create_dataset("cmv_dir_sigma", (0,),maxshape=(None,), compression="gzip")


        grp = f.create_group("meta")
        dset = grp.create_dataset('cloud_cover',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Cloud cover as the ratio of non-cloudy pixels to cloudy-pixels of the original image [0-1]"
        dset = grp.create_dataset('cloud_cover_stretched',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Cloud cover as the ratio of non-cloudy pixels to cloudy-pixels of the original image [0-1] in the equidistant stretched image"
        dset = grp.create_dataset('sun_azimuth',(0,),maxshape=(None,), compression="gzip")
        dset = grp.create_dataset('sun_zenith',(0,), maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Sun zenith angle [degrees]"
        dset = grp.create_dataset('cloud_base_height',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Cloud base height as used for cloud mapping and cloud motion vector calculations, derived from ceilometer data ( first cloud layer )"
        dset = grp.create_dataset('ncmv',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Number of valid cloud motion vectors used for averaging"
        dset = grp.create_dataset('img_qf',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Quality flag for image quality (OK, BAD)"
        dset = grp.create_dataset('lens_clear',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "Flag if lens is clear or not (e.g. birds)"
        if ini.cloud_class_apply:
            dset = grp.create_dataset('CloudClass',(0,),maxshape=(None,), compression="gzip")
            dset.attrs['Description'] = "Cloud class (1-7) from image cloud classification"
            dset = grp.create_dataset('CloudClassProb',(0,7),maxshape=(None,7), compression="gzip")
            dset.attrs['Description'] = "Cloud class probability"


        grp = f.create_group('image')
        #dset = grp.create_dataset('sunspot',(0,),maxshape=(None,), compression="gzip")
        #dset.attrs['Description'] = "percentage of pixels in sunspot area (<=3 degrees) which are saturated in hue channel"
        dset = grp.create_dataset('gray',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "gray level coefficient"


        if ini.features:
            for i in "0.5","1.5","3", "5","7","10","15","20":
                n = str(i)
                dset = grp.create_dataset('circ_'+n,(0,),maxshape=(None,), compression="gzip")
                dset.attrs['Description'] = "percentage of pixels in circumsolar area (<= " +n+" degrees) which are saturated in hue channel"

            for i in range(0,len(data['features'].vec)):
                dset = grp.create_dataset(data['features'].names[i],(0,), maxshape=(None,), compression="gzip")

        if ini.radiation:

            grp = f.create_group("radiation")
            if not ini.live:
                dset = grp.create_dataset('ghi',(0,),maxshape=(None,), compression="gzip")
                dset.attrs['Description'] = "measured global horizontal irradiance"
                dset = grp.create_dataset('dhi',(0,),maxshape=(None,), compression="gzip")
                dset.attrs['Description'] = "measured diffuse horizontal irradiance"
                dset = grp.create_dataset('dni',(0,),maxshape=(None,), compression="gzip")
                dset.attrs['Description'] = "measured direct normal irradiance"

        dset = grp.create_dataset('cls_ghi',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "SOLIS clear sky global horizontal irradiance"
        dset = grp.create_dataset('cls_dhi',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "SOLIS clear sky diffuse horizontal irradiance"
        dset = grp.create_dataset('cls_dni',(0,),maxshape=(None,), compression="gzip")
        dset.attrs['Description'] = "SOLIS clear sky direct normal irradiance"

        f.attrs['Title'] =  "Sky Imager Based Irradiance Forecasts"
        f.attrs['Institution'] = "University of Oldenburg, Energy Meteorology"
        f.attrs['Contact_Person'] = "Thomas Schmidt (Email: t.schmidt@uni-oldenburg.de)"
        f.attrs['History'] = "Processed with Sky Imager Analysis and Forecast Tool Version" + data['version']
        f.attrs['Location'] = "Position of skyimager: Latitude: " + str(ini.lat0) + " Longitude: " + str(ini.lon0)
        f.attrs['Processing_Date'] = "File created on " + str(datetime.utcnow())
        f.attrs['Author'] =  "Thomas Schmidt (Email: t.schmidt@uni-oldenburg.de)"

    else:

        flag = True
        while flag:
            try:
                f = h5py.File(filename,'a')
                flag = False
            except OSError:
                time.sleep(1)
                continue

    #############
    # Write data
    #############

    dset = f['time']
    dset = hdfresize(dset,add=1)
    pos = dset.shape[0] - 1
    dset[pos] = tstamp

    if ini.fcst_flag:
        dset = f['forecast/fghi']
        dset = hdfresize(dset,add=1)
        for i in range(0,len(data['fcst'])):
            dset[pos,:,i] = data['fcst'][i].fghi[:]
        dset = f['forecast/fdhi']
        dset = hdfresize(dset,add=1)
        for i in range(0,len(data['fcst'])):
            dset[pos,:,i] = data['fcst'][i].fdhi[:]
        dset = f['forecast/fdni']
        dset = hdfresize(dset,add=1)
        for i in range(0,len(data['fcst'])):
            dset[pos,:,i] = data['fcst'][i].fdni[:]
        if ini.radiation and not ini.live:
            dset = f['forecast/mghi']
            dset = hdfresize(dset,add=1)
            for i in range(0,len(data['fcst'])):
                dset[pos,:,i] = data['fcst'][i].fmeas[:]
    
        # clear sky and cap error not for all stations
        dset = f['forecast/cghi']
        dset = hdfresize(dset,add=1)
        dset[pos,:] = data['csk'][:]


    if ini.flow_flag:
        dset = f['cmv/cmv_u']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cmv'].mean_u
        dset = f['cmv/cmv_v']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cmv'].mean_v
        dset = f['cmv/cmv_dir_sigma']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cmv'].sdirection[-1]
        dset = f['cmv/cmv_spd_sigma']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cmv'].sspeed[-1]


    if ini.features:
        for i in range(0,len(data['features'].vec)):
            dset = f['image/' + data['features'].names[i]]
            dset = hdfresize(dset,add=1)
            dset[pos] = data['features'].vec[i]




    if ini.radiation:
        if not ini.live:
            # Radiation measurements
            dset = f['radiation/ghi']
            dset = hdfresize(dset,add=1)
            dset[pos] = data['ghi']
            try:
                dset = f['radiation/dni']
                dset = hdfresize(dset,add=1)
                dset[pos] = data['dni']
            except:
                pass
            try:
                dset = f['radiation/dhi']
                dset = hdfresize(dset,add=1)
                dset[pos] = data['dhi']
            except:
                pass
        dset = f['radiation/cls_ghi']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cls_ghi']
        dset = f['radiation/cls_dni']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cls_dni']
        dset = f['radiation/cls_dhi']
        dset = hdfresize(dset,add=1)
        dset[pos] = data['cls_dhi']


    for name in data['meta'].keys():
        try:
            dset = f['meta/' + name]
            dset = hdfresize(dset,add=1)
            dset[pos] = data['meta'][name]
        except:
            pass

    dset = f['meta/cloud_base_height']
    dset = hdfresize(dset,add=1)
    dset[pos] = data['meta']['cbh']

    dset = f['meta/lens_clear']
    dset = hdfresize(dset,add=1)
    dset[pos] = data['meta']['lens']

    dset = f['meta/img_qf']
    dset = hdfresize(dset,add=1)
    dset[pos] = data['meta']['img_qc']

    #dset = f['image/sunspot']
    #dset = hdfresize(dset,add=1)
    #dset[pos] = data['meta']['sunspot']

    dset = f['image/gray']
    dset = hdfresize(dset,add=1)
    dset[pos] = data['meta']['gray']


    if ini.features:
        for key in data['meta_rad']['circ']:
            dset = f['image/' + 'circ_' + str(key)]
            dset = hdfresize(dset,add=1)
            dset[pos] = data['meta_rad']['circ'][key]

    del dset


    f.close()
