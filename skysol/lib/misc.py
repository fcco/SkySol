import sys
from numpy import cos, sin, tan, degrees, radians, arctan2, pi, arcsin, log
import numpy as np
from datetime import datetime
import calendar
import os.path
import pytz

def date_to_julian_day(my_date):
    """Returns the Julian day number of a date."""
    a = (14 - my_date.month)//12
    y = my_date.year + 4800 - a
    m = my_date.month + 12*a - 3
    return my_date.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def check_settings(ini):
    """ Validate the settings.ini """

    print("Check Settings...")

    if ini.mode == 1 and ini.flow_flag:
        print("Mode = Single Pictures and Optical Flow activated -> Switch Optical Flow off!")
        ini.flow_flag = False
        ini.write_cmv = False
        ini.write_hdf5 = False

    if ini.mode == 1:
        print("Mode = Single Pictures and Forecast Flag activated -> Switch Forecast off!")
        ini.fcst_flag = False

    if ini.mode == 1:
        ini.outdir = ini.outdir + os.sep + os.path.basename(ini.imgpath)
        print("I changed output directory for analyses graphics to" + ini.outdir + " !")


    if ini.csi_mode == "model" and ini.features == False:
        print("Irradiance model needs feature extraction -> Switch Feature Extraction on!")
        ini.features = True

    # Check directory existence
    if not os.path.exists(ini.outdir + os.sep + 'plots'):
        os.makedirs(ini.outdir + os.sep + 'plots')
    if not os.path.exists(ini.outdir + os.sep + 'tmp'):
        os.makedirs(ini.outdir + os.sep + 'tmp')
    if not os.path.exists(ini.rootdir + os.sep + 'data'):
        os.makedirs(ini.rootdir + os.sep + 'data')
    if ini.mode == 0:
        if not os.path.exists(ini.outdir + os.sep + 'plots' + os.sep + ini.datestr):
            os.makedirs(ini.outdir + os.sep + 'plots' + os.sep + ini.datestr)

    if ini.log_flag and not os.path.exists(ini.outdir + os.sep + 'log'):
            os.makedirs(ini.outdir + os.sep + 'log')
            ini.log_file = ini.outdir + os.sep + 'log' + os.sep +  \
                datetime.datetime.strftime(datetime.datetime.utcnow(), \
                "%Y%m%d_%H%M%S") + '_' + ini.datestr + '.log'
            sys.stdout = open(ini.logfile,'a')

    if ini.write_cmv and not os.path.exists(os.path.dirname(ini.outdir + os.sep + 'cmv')):
        os.makedirs(os.path.dirname(ini.outdir + os.sep + 'cmv'))

    if ini.features and not os.path.exists(ini.outdir + os.sep + 'meta'):
        os.makedirs(ini.outdir + os.sep + 'meta')

    if ini.tz != "UTC":
        ltz = pytz.timezone(ini.tz)
        ini.locdate = pytz.UTC.localize(ini.actdate,is_dst=False).astimezone(ltz)
        ini.locdate_str = datetime.strftime(ini.locdate, "%Y%m%d")
    else:
        ini.locdate = ini.actdate
        ini.locdate_str = ini.datestr

    ini.hdffile = datetime.strftime(ini.locdate,ini.hdffile)
    ini.outfile = datetime.strftime(ini.locdate,ini.outfile)

    if ini.write_meta and not os.path.exists(os.path.dirname(ini.outfile)):
        os.makedirs(os.path.dirname(ini.outfile))

    if ini.write_hdf5 and not os.path.exists(os.path.dirname(ini.hdffile)):
        os.makedirs(os.path.dirname(ini.hdffile))

    print("...finished")
    print("")

    # Grid settings
    ini.grid_res_y, ini.grid_res_x = meter2degrees(ini.lon0,ini.lat0,ini.x_res,ini.y_res)
    ini.lat_min, ini.lat_max, ini.lon_min, ini.lon_max = \
        gridBoundaries(ini.lat0, ini.lon0, ini.x_res, ini.y_res, ini.grid_size)



def bearing(lat1,lon1,lat2,lon2):
    """
    In general, your current heading will vary as you follow a great circle
    path (orthodrome); the final heading will differ from the initial heading
    by varying degrees according to distance and latitude (if you were to go
    from say 35N,45E ( Baghdad) to 35N,135E ( Osaka), you would start on
    a heading of 60 and end up on a heading of 120!).
    This formula is for the initial bearing (sometimes referred to as forward azimuth)
    which if followed in a straight line along a great-circle arc will take you
    from the start point to the end point:1 Calculate bearing
    """
    #theta = arctan2( sin(lon2-lon1) * cos(lat2), \
    # cos(lat1) * sin(lat2)-sin(lat1) * cos(lat2) * cos(lon2-lon1))

    dtheta = log(tan(radians(lat2)/2 + pi/4) / tan( radians(lat1) / 2 + pi/4))
    dlon = radians(lon1)-radians(lon2)
    theta  = np.arctan2(dlon,dtheta)

    return theta + np.pi

def coordDist(lat0,lon0,lat1,lon1):
    # Calculate distance between start and end points in km
    lat = (lat0 + lat1) / 2 * pi/180.
    dx = 111.3 * cos(lat) * (lon1 - lon0)
    dy = 111.3 * (lat1 - lat0)
    distance = np.sqrt(dx * dx + dy * dy)
    return distance



def coordFromDir(lat0,lon0,angle,d,R=6372.795477598):
    # distance in km
    # angle in radians
    # R is earth radius
    lat = arcsin( sin(radians(lat0))*cos(d/R) + \
        cos(radians(lat0))*sin(d/R)*cos(angle) )
    lon = radians(lon0) + arctan2(sin(angle)*sin(d/R)*cos(radians(lat0)), \
        cos(d/R)-sin(radians(lat0))*sin(lat))

    return degrees(lat),degrees(lon)


def polar2grid(r, alpha, size, xres, yres):
    """
    Apply a polar to cartesian transformation with respect to defined grid

    :params r: distance in kilometer
    :type r: float
    :params alpha: angle (radians) from north
    :type alpha: float
    :params size: grid size
    :type size: float or int
    :params xres, yres: grid resolution in x/y direction in meter
    :type xres,yres: float or int
    """


    y = (1000 * r * np.sin(alpha) / yres) + size / 2.
    x = (1000 * r * np.cos(alpha) / xres) + size / 2.

    return y, x


def dump(obj):
    for attr in dir(obj):
        print("obj.%s = %s" % (attr, getattr(obj, attr)))
    print(" ")


def gridRes(obj):
    # Calculate grid resolution
    latRes = coordDist(obj.lat0,obj.lon0,obj.lat0+obj.grid_res_y,obj.lon0) *1000
    lonRes = coordDist(obj.lat0,obj.lon0,obj.lat0,obj.lon0+obj.grid_res_x) * 1000
    return latRes, lonRes


def meter2degrees(lon0,lat0,xres,yres):

    # x-direction
    lat1, lon1 = coordFromDir(lat0,lon0,pi/2.,xres/1000.)
    lonRes = lon1 - lon0

    # y-direction
    lat1, lon1 = coordFromDir(lat0,lon0,0,yres/1000.)
    latRes = lat1 - lat0

    return latRes, lonRes


def latlon2grid(latRes,lonRes,speed,angle):

    # reconvert speed and direction to u/v components
    u = -1. * speed * sin(angle)
    v = -1. * speed * cos(angle)

    # result: grid cells per second in north and east direction
    ull = u / float(latRes)
    vll = v / float(lonRes)
    return ull, vll


def grid2latlon(lat0,lon0,xres,yres,size,y,x):
    """
    Convert grid coordinates in lat lon

    input:
    lat0, lon0: coordinates of central point (camera location)
    xres, xres: grid resolution in meter
    size: grid size in cells
    y: y - grid coordinate ( west-> east or left -> right)
    x: x - grid coordinate ( north -> south or top -> bottom)

    returns: lat, lon from given grid coordinate

    """
    nx = (size-x) - size/2
    ny = y - size/2
    alpha = np.arctan2(ny,nx)
    dist = np.sqrt((xres*nx)**2 + (yres*ny)**2) / 1000

    return coordFromDir(lat0,lon0,alpha,dist)


def gridBoundaries(lat, lon, xres, yres, size):
    """
    Compute minimum and maximum longitude and latitude of the grid
    """
    # grid size in m
    lat_max, _ = coordFromDir(lat, lon, 0, yres*size/1000/2)
    _, lon_max = coordFromDir(lat, lon, pi/2, xres*size/1000/2)
    lat_min, _ = coordFromDir(lat, lon, pi, xres*size/1000/2)
    _, lon_min = coordFromDir(lat, lon, 3/2.*pi, yres*size/1000/2)

    return lat_min, lat_max, lon_min, lon_max


def distance2ImCenter(lat0,lon0,lat1,lon1,cldhgt,sza,saz):
    d1 = float(cldhgt) * tan(sza)
    latImCenter, lonImCenter = coordFromDir(lat0,lon0,saz,d1)
    return coordDist(lat0,lon0,latImCenter,lonImCenter)


def shadow_dist(hgt,theta):
    dist = float(hgt) * tan(theta)
    return dist

def sunpixelangle(theta,phi,sza,saz):
    # Angular distance  (Sun-Pixel-Angle SPA) to the sun for each pixel
    spa = np.sqrt(np.square(theta - sza) + \
        np.multiply(np.square(np.sin(sza)) , np.square(phi - saz ) ) )

    return spa


def get_index(timearr,tstep):

    # numeric timestamp and datetime array
    if type(timearr) is datetime and type(tstep).__module__ == np.__name__:
        tstep = datetime.utcfromtimestamp(tstep)

    # datetime timestamp and numeric array
    if type(timearr).__module__ == np.__name__ and type(tstep) == datetime:
        tstep = int(tstep.timestamp())

    # list
    if type(timearr) is type(list()):
        if type(tstep) == datetime and type(timearr[0]) != datetime: tstep = tstep.timestamp()
        try:
            index = timearr.index(tstep)
        except ValueError:
            print("No measurements for timestep", tstep, "found -> Skip")
            index = -1

    # datetime instance
    elif type(timearr) is datetime:
        try:
            index = np.where(timearr == tstep)
            return index
        except IndexError:
            print("No measurements for timestep", tstep, "found -> Skip")
            index = -1

    # numpy numeric timestamp
    elif type(timearr).__module__ == np.__name__:
        try:
            index = np.where(timearr == tstep)[0][0]
            return index
        except IndexError:
            print("No data for timestep", datetime.utcfromtimestamp(tstep), "found -> Skip")
            index = -1
    else:
        print("Wrong data type for time array!!!")
        index = -1

    return index


def pix_res(cx,cy,theta,cbh):

    pixres = cbh * (np.tan(theta[cy,cx+1:]) - \
        np.tan(theta[cy,cx:-1]))

    return pixres, theta[cy,cx+1:]




def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km * 1000



def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result


class args:

    def __init__(self, argv):

        self.mode = 'live'
        self.indate = ""
        self.edate = ""
        self.sdate = ""
        self.imgpath = ""
        self.isclear = False

        self.inArgHandler(argv)

    def inArgHandler(self, argv):

        import getopt, sys

        try:
            opts, args = getopt.getopt(argv, "d:m:s:e:p:h", ["date=", "mode=", "sdate=", "edate=", "isclear="])
        except getopt.GetoptError as er:
            print(str(er) + 'Type `python main.py -d <YYYYMMDD> -m <mode> -s <YYYYMMDD_HHMMSS> -e <YYYYMMDD_HHMMSS> -p <PATH_TO_IMAGES> --isclear=<bool>`')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('main.py -d <YYYYMMDD> -m <mode> -s <YYYYMMDD_HHMMSS> -e <YYYYMMDD_HHMMSS> -p <PATH_TO_IMAGES> --isclear=<bool>')
                sys.exit()
            elif opt in ("-d", "--date"):
                self.indate = datetime.strptime(arg, "%Y%m%d")
            elif opt in ("-s", "--sdate"):
                self.sdate = datetime.strptime(arg, "%Y%m%d_%H%M%S")
            elif opt in ("-e", "--edate"):
                self.edate = datetime.strptime(arg, "%Y%m%d_%H%M%S")
            elif opt in ("-m", "--mode"):
                self.mode = arg
            elif opt in ("--isclear"):
                self.isclear = eval(arg)
            elif opt in ("-p"):
                self.imgpath = arg
                print('Changed image directory to ' + str(arg))
                self.mode = "single"
        if self.sdate == "": self.sdate = self.indate
        if self.indate == "" and self.sdate != "":
            self.indate = self.sdate
            self.mode = "archive"
        if self.indate != "":
            self.mode = "archive"
        elif self.indate == "":
            self.indate = datetime.utcnow()
