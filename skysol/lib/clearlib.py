'''
Created on 29.08.2014

@author: thomas.schmidt
'''
import os
import h5py
import numpy as np

def hdfresize(dset,add=1):
    """ Resize hdf5 dataset """
    shape = list(dset.shape)
    shape[0] = shape[0] + add
    dset.resize(shape)
    return dset

class csl:

    def __init__(self, filename, read=True):
        self.cslfile = filename

        if not read:
          self.time = []
          self.az = []
          self.theta = []
          self.csl_flag = False
          return

        # Read in CSL library
        try:
            with h5py.File(self.cslfile, 'r') as f:
                az = f['azimuth'][:]
                self.az = np.radians(az)
                theta = f['zenith'][:]
                self.theta = np.radians(theta)
                self.time = f['time'][:]
                self.csl_flag = True
        except:
            print('Could not read CSL file %s. Switch off CSL correction.' % self.cslfile)
            self.csl_flag = False




    def read_csl(self, tstamp, saz, sza):
        """
        Get the timestamp of the current reference clear sky image.

        Clear sky images are used, if the sun-pixel-angle is less than 5 degree.
        If there are more than one image available, take the most up to date.
        """

        elev = np.pi/2 - self.theta
        sea = np.pi/2 - sza
        spa = np.arccos(np.sin(sea)*np.sin(elev) + np.cos(sea) * \
             np.cos(elev) * np.cos(self.az-saz) )
        try:
            elev = np.pi/2 - self.theta
            sea = np.pi/2 - sza
            spa = np.arccos(np.sin(sea)*np.sin(elev) + np.cos(sea) * \
                 np.cos(elev) * np.cos(self.az-saz) )
        except AttributeError:
            return ""

        # library value should be closer than 5 degrees to current sun position
        ind = (spa < np.radians(5))
        # library value should be younger than 30 days
        tdiff = tstamp - self.time[ind]
        tind =  np.abs(tdiff) < 30*86400
        t = np.abs(tdiff) >= 0
        if np.sum(tind[t]) > 0:
            # find latest matching value (in time)
            tdiff = np.abs(tdiff)
            ti = np.where(np.min(tdiff[t]) == tdiff)[0]
            # based on latest matching value (for angular distances < 5),
            # look for the reference with the closest sun angular distance
            # (consider all references, that are not older than one day)
            sind = np.abs(self.time[ind][ti[0]] - self.time[ind]) < 86400
            ti = np.where(np.min(spa[ind][sind]) == spa[ind])[0]
            return self.time[ind][ti[0]]
        else:
            return -1


    def write_db(self, tstamp, sza, saz):
        """
        Add an entry to the Clear Sky Library
        """
        ret = False

        sza = np.degrees(sza)
        saz = np.degrees(saz)

        if not os.path.exists(self.cslfile):

            # Create CSL if run for the first time
            with h5py.File(self.cslfile, 'w') as f:
                dset = f.create_dataset('zenith',maxshape=(None,),data=[sza])
                dset.attrs['unit'] = 'degrees'
                dset = f.create_dataset('azimuth',maxshape=(None,),data=[saz])
                dset.attrs['unit'] = 'degrees'
                dset = f.create_dataset('time',maxshape=(None,),data=[tstamp])
                dset.attrs['unit'] = 'UTC'
                dset.attrs['description'] = 'Unix Timestamp'
                self.time = np.array([tstamp])
                self.theta = np.array([np.radians(sza)])
                self.az = np.array([np.radians(saz)])

                ret = True

        else:

            try:

                with h5py.File(self.cslfile,'a') as f:

                    ind = np.where(tstamp == self.time)[0]

                    if len(ind) == 0:

                        dset = f['zenith']
                        dset = hdfresize(dset, add=1)
                        dset[-1] = sza

                        dset = f['azimuth']
                        dset = hdfresize(dset, add=1)
                        dset[-1] = saz

                        dset = f['time']
                        dset = hdfresize(dset, add=1)
                        dset[-1] = tstamp

                        self.time = np.append(self.time, tstamp)
                        self.theta = np.append(self.theta, np.radians(sza))
                        self.az = np.append(self.az, np.radians(saz))

                        ret = True
            except OSError as e:
                print(e)
                pass

        self.csl_flag = True

        return ret
