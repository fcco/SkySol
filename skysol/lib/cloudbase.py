"""
Includes main class for cloud base information
Includes functions for import of cloud height data from different sources
"""
from __future__ import print_function
from datetime import datetime, timedelta
import time
import numpy as np
import os

class cloudbase:

    def __init__(self,ini):
        self.raw = []  # raw = backscatter rawsignal
        self.range = []
        self.cbh = np.nan # The latest cloud base height used for all computations
        self.cbh_raw = [] # List of all loaded CBHs from day or period
        self.time = [] # Epoch timestamps
        self.dates = [] # Datetime timestamps
        self.wspd_now = None
        self.name = ini.rootdir + os.sep + 'tmp' + os.sep + 'clouds'
        self.file_flag = True

        source = ini.cbh_source

        if ini.cbh_flag:

            st = time.time()
            if source == "metar":
                self.cbh_raw, self.time, self.dates = self.read_metar(ini)

            elif source == "":
                self.cbh = ini.def_cbh
                self.file_flag = False
            else:
                "CBH source " + source + " invalid! Specify another source or keep it empty"
        else:
            self.cbh = ini.def_cbh
            self.file_flag = False

    def read_metar(self, ini):

        self.tem =  []
        self.wdir = []
        self.wspd = []
        self.cbh_type = []
        self.cbh_type2 = []
        self.cbh_type3 = []
        self.cbh_raw2 = []
        self.cbh_raw3 = []

        for i in range(0,2):

            dtstr = datetime.strftime(ini.actdate - timedelta(days=i),"%Y%m%d.metar")
            filename = ini.cldfile + os.sep + dtstr

            f = open(filename)

            for row in f:

                ls = row.split(' ')
                if ls[0] == "ICAO": continue
                d = datetime.strptime(ls[1]+' '+ls[2],"%Y-%m-%d %H:%M:%S")
                try:
                    wspd = float(ls[6])
                except:
                    continue
                tem = float(ls[4])
                self.tem.append(tem)
                self.dates.append(d)
                self.time.append(np.int64((ls[3])))
                self.wdir.append(float(ls[5]))
                self.wspd.append(wspd)
                try:
                    self.cbh_raw.append(float(ls[8]))
                    self.cbh_type.append(ls[7])
                except IndexError:
                    self.cbh_raw.append(np.nan)
                    self.cbh_type.append("")
                try:
                    self.cbh_raw2.append(float(ls[10]))
                    self.cbh_type2.append(ls[9])
                except IndexError:
                    self.cbh_raw2.append(np.nan)
                    self.cbh_type2.append("")
                try:
                    self.cbh_raw3.append(float(ls[12]))
                    self.cbh_type3.append(ls[11])
                except IndexError:
                    self.cbh_raw3.append(np.nan)
                    self.cbh_type3.append("")

        return np.array(self.cbh_raw), np.array(self.time), np.array(self.dates)


    def current(self, actnum, oldhgt, defcldhgt):
        """ Get the current cloud base height """

        prof=[]
        # keep latest cbh
        cldhgt = oldhgt

        timediff = np.abs(self.time - actnum)
        closest = np.nanmin(timediff)
        if closest < 1900:
            ind = np.argmin(timediff)
            cldhgt = self.cbh_raw[ind]
        else:
            cldhgt = oldhgt

        if cldhgt < 100:
            cldhgt = defcldhgt
        if np.isnan(cldhgt):
            cldhgt = defcldhgt

        self.cbh = cldhgt

        return cldhgt
