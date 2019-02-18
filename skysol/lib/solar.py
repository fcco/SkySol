# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
# title	       : solar.py
# description  : computation of solar geometry (zenith and azimuth angle)
# version      : 0.1
# usage		  : solar.compute( dates, lat, lon )
# notes		  :
# python_version   : 3.5

Created on Wed Jul 13 14:49:47 2016

@author: thomas.schmidt
#==============================================================================
"""
from pvlib import solarposition
from pandas import DatetimeIndex
from datetime import datetime
import numpy as np
from numpy import pi, cos, sin, radians, degrees, arcsin, arccos


def compute(dates,lat,lon,method="spencer"):
    """
    Call this function to compute solar geometry/position for given timestamp and
    location. Different algorithms are implemented.

    .. note:

    If packages (pvlib or pysolargeometry) are not installed, choose another method.

    Try to use python-datetime lists or arrays as input.
    Some methods have issues with single instances or other time
    formats ( epoch / julian / datetime64 /....)

    Parameters
    -----------

    :params dates: list/array of datetime instances
    :params type: list or array

    :params lat: geographical latitude of location
    :params type: float

    :params lon: geographical longitude of location
    :params type: float

    :params method: algorithm to be used, default is solar geometry2 in package
        pysolargeometry
    :params type: string

    Returns
    --------

    :returns: dictionary with at least 'zenith' and 'azimuth' angles in radians

    """
    if method == "sg1" or method == "sg2":

        from pysolargeometry import util
        return util.compute(dates,lat,lon,method=method)

    elif method == "ephemeris":

        times = DatetimeIndex(dates)
        sun = {}
        solar = solarposition.get_solarposition(times,lat,lon,method="ephemeris")

        sun['zenith'] = radians(90 - solar['elevation'])
        sun['azimuth'] = radians(solar['azimuth'])

        return sun

    elif method == "nrel_c":

        times = DatetimeIndex(dates)
        sun = {}
        solar = solarposition.get_solarposition(times,lat,lon,method="nrel_c")
        print(solar)
        sun['zenith'] = radians(90 - solar['elevation'])
        sun['azimuth'] = radians(solar['azimuth'])

        return sun

    elif method == "spencer":

        sun = {}
        dates = [ d.replace(tzinfo=None) for d in dates ]
        solar = solar_data(dates,lat,lon)

        sun['zenith'] = radians(solar['zenith'])
        sun['azimuth'] = radians(solar['azimuth'])

        return sun



def solar_data(dates, lat, lon):
    """
    Simple solar position algorithm.

    :param dates: list of datetime objects (must be in UTC)
    :type dates: list

    :param lat: latitude (degrees)
    :type lat: float

    :param lon: longitude (degrees)
    :type lat: float

    :returns: dictionary with 'zenith' - solar zenith angle (degrees)
                             'azimuth' - solar azimuth angle (degrees)
                             'I_Ext' - extraterrestrial radiation (W/m^2)

    .. note::

        Formulas from Spencer (1972) and can be also found in "Solar energy
        fundamentals and modeling techniques" from Zekai Sen
    """

    dt_1600 = datetime(1600, 1, 1)

    # Compute julian dates relative to 1600-01-01 00:00.
    julians_1600 = (np.array([(dt - dt_1600).total_seconds()
            for dt in dates]) / (24 * 60 * 60))

    lat = radians(lat)
    lon = radians(lon)

    # Compute gamma angle from year offset and calculate some intermediates.
    Gamma = 2. * pi * (julians_1600 % 365.2425) / 365.2425
    cos_gamma = cos(Gamma), cos(Gamma * 2), cos(Gamma * 3)
    sin_gamma = sin(Gamma), sin(Gamma * 2), sin(Gamma * 3)
    DayTime = (julians_1600 % 1) * 24

    # Eccentricity: correction factor of the earth's orbit.
    ENull = (1.00011 + 0.034221 * cos_gamma[0] + 0.001280 * sin_gamma[0] +
            0.000719 * cos_gamma[1] + 0.000077 * sin_gamma[1])

    # Declination.
    Declination = (0.006918 - 0.399912 * cos_gamma[0] +
            0.070257 * sin_gamma[0] -
            0.006758 * cos_gamma[1] + 0.000907 * sin_gamma[1] -
            0.002697 * cos_gamma[2] + 0.001480 * sin_gamma[2])

    # Equation of time (difference between standard time and solar time).
    TimeGl = (0.000075 + 0.001868 * cos_gamma[0] - 0.032077 * sin_gamma[0] -
            0.014615 * cos_gamma[1]  - 0.040849 * sin_gamma[1]) * 229.18

    # True local time    .
    Tlt = (DayTime + degrees(lon) / 15 + TimeGl / 60) % 24 - 12

    # Calculate sun elevation.
    SinSunElevation = (sin(Declination) * sin(lat) +
            cos(Declination) * cos(lat) * cos(radians(Tlt * 15)))

    # Compute the sun's elevation and zenith angle.
    el = arcsin(SinSunElevation)
    zenith = pi / 2 - el

    # Compute the sun's azimuth angle.
    y = -(sin(lat) * sin(el) - sin(Declination)) / (cos(lat) * cos(el))
    azimuth = arccos(y)

    # Convert azimuth angle from 0-pi to 0-2pi.
    Tlt_filter = 0 <= Tlt
    azimuth[Tlt_filter] = 2 * pi - azimuth[Tlt_filter]

    # Calculate the extraterrestrial radiation.
    INull = np.max(1360.8 * SinSunElevation * ENull, 0)

    return {
        'zenith': np.degrees(zenith),
        'azimuth': np.degrees(azimuth),
        'declination': Declination,
        'I_ext': INull,
        'eccentricity': ENull,
    }
