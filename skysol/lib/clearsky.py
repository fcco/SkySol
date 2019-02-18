import calendar
import time
import numpy as np


class clearsky:

    def __init__(self,dates,lat,lon,source="pvlib", model="ineichen", quiet=True):

        """
        Class for clear sky irradiance data computed based on specific algorithms

        Parameters
        ----------

        :param dates: list/vector of datetime objects
        :param lat: float, locations geographical latitude (degrees)
        :param lon: float, locations geographical longitude (degrees)
        :param source: string, methodology used to calculate clear sky irradiance

        Notes
        ------

        Source: Currently, two algorithms are supported "pypvsim" and "pvlib".

        Returns
        --------

        The functions return clear sky irradiance for given timestamps and location
        3 components GHI, DHI and DNI are returned

        """
        st = time.time()

        self.time = np.array([calendar.timegm(dt.utctimetuple()) for dt in dates])
        self.dates = dates

        if source == "pvlib":

            self.ghi, self.dhi, self.dni = self.pvlib(dates,lat,lon, \
                method=model)

        else:
            print("Wrong method chosen: %s Read function help for supported methods" % \
                source)


        if not quiet: print('finished in',round(time.time()-st,1))



    def pvlib(self, dates, lat, lon, method="ineichen"):

        from pvlib import clearsky, solarposition
        from pvlib.location import Location
        from pandas import DatetimeIndex

        times = DatetimeIndex(dates)

        loc = Location(lat, lon)
        clear = loc.get_clearsky(times, model=method)

        return clear['ghi'], clear['dhi'], clear['dni']
