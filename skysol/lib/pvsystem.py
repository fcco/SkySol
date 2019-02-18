# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
# title	        :
# description	  :
# version		  :
# usage		  :
# notes		  :
# python_version   :

Created on Tue Jul 12 18:02:03 2016

@author: thomas.schmidt
#==============================================================================

"""
from pvlib import pvsystem, irradiance, atmosphere
from pandas import DatetimeIndex
import numpy as np
from numpy import degrees as deg
from numpy import radians as rad

class pvsys:

    def __init__(self, method="pypvsim", surface_tilt=0, surface_azimuth=180, \
        pmax=0, albedo=.25, surface_type=None, module=None, module_parameters=None, \
        modules_per_string=1, strings_per_inverter=1, inverter=None, inverter_parameters=None, \
        racking_model='open_rack_cell_glassback', **kwargs):

        self.tilt = surface_tilt
        self.azimuth = surface_azimuth
        self.albedo = albedo
        self.pmax = pmax
        self.racking_model = 'open_rack_cell_glassback'
        self.method = method

        if "lat" in kwargs:
            self.lat = kwargs['lat']
        if "lon" in kwargs:
            self.lon = kwargs['lon']
        if "dates" in kwargs:
            self.dates = kwargs['dates']

        if self.method == "pvlib":

            self.params = {}
            self.params['pdc0'] = pmax # DC nameplate rating
            self.params['gamma_pdc'] = 0.0044 # temperature coefficient. Typically in units of 1/C.
            self.params['eta_inv'] = 0.97 # inverter efficiency
            self.params['eta_inv_ref'] = 0.90 # reference inverter efficiency

            self.syst = pvsystem.PVSystem(surface_tilt=self.tilt, surface_azimuth=self.azimuth, \
                albedo=self.albedo, surface_type=surface_type, module=module, module_parameters=self.params, \
                modules_per_string=1, strings_per_inverter=1, inverter=None, inverter_parameters=None, \
                racking_model=self.racking_model)





    def get_aoi(self, sun_zenith, sun_azimuth):

        if self.method == "pvlib":
            self.aoi = rad(self.syst.get_aoi(deg(sun_zenith), deg(sun_azimuth)))




    def get_poa(self, dates, ghi, dhi, dni, sun_zenith, sun_azimuth, dni_extra=None,
                airmass=None, model="haydavies"):


        if self.method == "pvlib":

            sun_zenith = deg(sun_zenith)
            sun_azimuth = deg(sun_azimuth)

            if dni_extra is None:
                dni_extra = irradiance.extraradiation(DatetimeIndex(dates))
            if airmass is None:
                 airmass = atmosphere.relativeairmass(sun_zenith)

            poa = irradiance.total_irrad(self.tilt, self.azimuth, sun_zenith, sun_azimuth,
                dni, ghi, dhi, dni_extra=dni_extra, airmass=airmass,
                model=model,albedo=self.albedo)

            self.poa = poa['poa_global']


    def compute_power(self):

        if self.method == "pvlib":

            self.dc = self.syst.pvwatts_dc(self.poa, self.tcell)
            #self.ac = self.syst.pvwatts_ac(self.dc)




    def celltemp(self,irrad, wind=None, temp=None):
        """
        Compute cell temperature based on SAPM model
        You can provide Wind and Ambient air temperature if available. if not
        constant 25 deg C and 0 m/s wind are assumed
        """

        if len(temp) == 0:

            temp = np.zeros_like(irrad)
            temp[:] = 25.

        if len(wind) == 0:

            wind = np.zeros_like(irrad)
            wind[:] = 0.

        if self.method == "pvlib":

            temp = self.syst.sapm_celltemp(irrad,wind,temp)
            self.tcell = temp['temp_cell'].values
            self.tmod = temp['temp_module'].values
