import numpy as np
from numpy import pi, arctan2

class cmv:

    def __init__(self,ini):

        self.npoints=0
        self.p0 = 0

        # feature image coordinates
        self.new_point = []
        self.old_point = []

        # feature cloud map coordinates
        self.old_map = []
        self.new_map = []

        # feature output plot coordinates
        self.x_new = 0.0
        self.x_old = 0.0
        self.y_new = 0.0
        self.y_old = 0.0

        # feature geographical coordinates
        self.lat_new = []
        self.lat_old = []
        self.lon_new = []
        self.lon_old = []

        # single u/v components
        self.u = []
        self.v = []

        # single u/v components from last image
        self.old_u = []
        self.old_v = []

        # List of global average speed and direction
        self.speed = []; self.sspeed = []
        self.direction = []; self.sdirection = []
        self.fcst_speed = []
        self.fcst_direction = []

        # Temporal smoothed CMV
        self.mean_speed = np.nan; self.std_speed = np.nan
        self.mean_direction = np.nan; self.std_direction = np.nan
        self.mean_u = np.nan; self.std_u = np.nan
        self.mean_v = np.nan; self.std_v = np.nan


        # List of global average u/v components
        self.us = []; self.std_us = []
        self.vs = []; self.std_vs = []


        # CMV quality flag
        self.flow_flag = [] # flag determined by optical flow algorithm
        self.flag = [] # own quality flag
        self.old_flag = [] # own quality flag from last image
        self.fcst_flag = [] # own quality flag from last image


        # Number of CMV for temporal smoothing/averaging
        self.tempavg = ini.cmv_temp_avg

        # seconds between current and last image
        self.lastimg = ini.rate

    def calcWind(self,ini):

        # Reset lists
        self.lat_new = []
        self.lat_old = []
        self.lon_new = []
        self.lon_old = []
        self.u = []
        self.v = []
        self.flag = []

        for j in range(0,self.npoints):

            # initialise lists with defaults
            self.flag.append(False)
            self.u.append(np.nan)
            self.v.append(np.nan)

            # Do not use points marked as invalid by optical flow algorithm
            if self.flow_flag[j] == 0: continue

            self.u[-1] = float( self.old_point_grid[j,0] - self.new_point_grid[j,0] ) / float(self.lastimg)
            self.v[-1] = float( self.new_point_grid[j,1] - self.old_point_grid[j,1] ) / float(self.lastimg)

            # convert velocity from pixel/s in meter/s with grid resolution
            self.u[-1] = self.u[-1] * ini.x_res
            self.v[-1] = self.v[-1] * ini.y_res

            self.flag[-1] = True





    def checkConsistency(self,t2target=False):
        """ Checks the quality motion vectors derived by optical flow algorithm

        This function flags vectors with a bad quality

        The quality checks are:

        check if ...
        1. Vector u or v-component is not 0.0
        2. The difference between the old and the new u/v-component is not greater than 1.5 m/s
        3. u/v is not NaN
        4. are points next to masked regions?
        """

        if t2target:
            minlimit = 0.0; maxlimit = 100; difflimit = 5
        else:
            minlimit = 0.2; maxlimit = 50; difflimit = 2.0

        speed = []; direction =  []; x = []; y = []

        # calculate new u and v

        if len(self.old_u) > 0:

            for i in range(0,self.npoints):

                if self.flag[i]:

                    if abs(self.u[i]) <= minlimit or \
                        abs(self.v[i]) <=  minlimit or \
                        abs(self.u[i]) > maxlimit or \
                        abs(self.v[i]) > maxlimit or \
                        abs(self.u[i] - self.old_u[i]) > difflimit or \
                        abs(self.v[i] - self.old_v[i]) > difflimit:

                        self.flag[i] = False
                    elif np.isnan(self.u[i]) or np.isnan(self.v[i]):
                        self.flag[i] = False
                    elif self.old_flag[i] == False:
                        self.flag[i] = False
                    else:
                        self.flag[i] = True
                        x.append(self.u[i]); y.append(self.v[i])
                        speed.append(np.sqrt(self.u[i] * self.u[i] + self.v[i] * self.v[i] ))
                        direction.append((arctan2(self.u[i],self.v[i])+2*pi) % (2*pi) )

        else:

            for i in range(0,self.npoints):

                if np.sqrt((self.new_point[i,0]-self.old_point[i,0])**2 + (self.new_point[i,1]-self.old_point[i,1])**2) < 2.0:
                    self.flag[i] = False

                elif abs(self.u[i]) < minlimit or \
                    abs(self.v[i]) < minlimit or \
                    abs(self.u[i]) > maxlimit or \
                    abs(self.v[i]) > maxlimit or \
                    np.isnan(self.u[i]) or \
                    np.isnan(self.v[i]) or \
                    self.flag[i] == False:

                    self.flag[i] = False
                else:
                    self.flag[i] = True

                    x.append(self.u[i]); y.append(self.v[i])
                    speed.append(np.sqrt(self.u[i] * self.u[i] + self.v[i] * self.v[i] ))
                    direction.append((arctan2(self.u[i],self.v[i])+2*pi) % (2*pi) )
                self.old_u.append(self.u[i]); self.old_v.append(self.v[i])
                self.old_flag.append(self.flag[i])



        # build global average CMV of all good quality single CMV
        if np.sum(self.flag) > 10:
            self.us.append(np.nanmean(x))     #  vector x-component in pixel/s
            self.vs.append(np.nanmean(y))   #  vector y-component in pixel/s
            self.std_us.append(np.nanstd(x))
            self.std_vs.append(np.nanstd(y))

            self.speed.append(np.nanmean(speed)) # speed in pixel/s or m/s
            self.direction.append((np.arctan2(np.nanmean(x),np.nanmean(y))+2*pi) % (2*pi) )# direction in radian
            self.sspeed.append(np.nanstd(speed)) # speed in pixel/s or m/s

            wdirdiff = np.array(np.abs(self.direction[-1] - (np.arctan2(x,y)+2*pi) % (2*pi) ))
            ind = wdirdiff > pi
            wdirdiff[ind] = np.abs(wdirdiff[ind] - 2*np.pi)

            self.sdirection.append(np.nanstd(wdirdiff))
            #self.sdirection.append(np.nanstd(2*pi + ((np.arctan2(x,y)) % (2*pi)) )) # direction in radian
        else:
            self.us.append(np.nan)     #  vector x-component in pixel/s
            self.vs.append(np.nan)   #  vector y-component in pixel/s
            self.std_us.append(np.nan)
            self.std_vs.append(np.nan)

            self.speed.append(np.nan) # speed in pixel/s or m/s
            self.direction.append(np.nan) # direction in radian
            self.sspeed.append(np.nan) # speed in pixel/s or m/s
            self.sdirection.append(np.nan) # direction in radian


        # reduce list size
        if len(self.speed) > 60:

            self.speed = self.speed[-60:]
            self.sspeed = self.sspeed[-60:]
            self.direction = self.direction[-60:]
            self.sdirection= self.sdirection[-60:]

            self.us = self.us[-60:]
            self.vs = self.vs[-60:]
            self.std_us = self.std_us[-60:]
            self.std_vs = self.std_vs[-60:]




    def plot_windbarbs(self,m):
        """
        Plot windbarbs on grid
        """

        for j in range(0,self.npoints):

            if self.flag[j]:

                self.x_old, self.y_old = m(self.lon_old[j], self.lat_old[j])
                self.x_new, self.y_new = m(self.lon_new[j], self.lat_new[j])

                m.quiver(self.x_old,self.y_old,self.x_new-self.x_old,self.y_new-self.y_old,width=0.002,color='r')



    def smoothWind(self):

        # build a temporal average CMV ( average over last x global CMVs )
        self.mean_u = np.nanmean(self.us[-self.tempavg:]); self.std_u = np.nanstd(self.us[-self.tempavg:])
        self.mean_v = np.nanmean(self.vs[-self.tempavg:]); self.std_v = np.nanstd(self.vs[-self.tempavg:])
        self.mean_speed = np.nanmean(self.speed[-self.tempavg:]); self.std_speed = np.nanstd(self.speed[-self.tempavg:])

        self.mean_direction = (np.arctan2(self.mean_u,self.mean_v)+2*pi) % (2*pi)

        wdirdiff = np.array(np.abs(self.mean_direction - (np.arctan2(self.us[-self.tempavg:],self.vs[-self.tempavg:])+2*pi) % (2*pi) ))
        ind = wdirdiff > pi
        wdirdiff[ind] = np.abs(wdirdiff[ind] - 2*np.pi)
        self.std_direction = np.nanstd(wdirdiff)
        #self.std_direction = np.nanstd(2*pi + ( (np.arctan2(self.us[-self.tempavg:],self.vs[-self.tempavg:])) % (2*np.pi) ))


def cart2sphere(x,y,z):
    """
    Coordinate transformation from cartesian (image) to
    spherical coordinates in 3d object space or
    real world on a unit sphere.

    Input: x, y, z cartesian coordinates

    zenith angle ( theta ) and azimuth angle ( phi )
    in radians

    Output: incidence angle ( theta ), azimuth angle ( phi, psi )
    and distance ( r )

    Note:
    The function should not be used for the computation of the
    incidence angle theta of an image pixel.
    The incidence angle should be computed from the fisheye projection.
    """
    from numpy import pi, arccos, arctan2

    # azimuth counterclockwise
    psi = arctan2(y,x)

    # azimuth clockwise counting from top of image
    phi = arctan2(-1*y,x) + pi/2.
    phi = ( phi + 2.*pi ) % (2.*pi) # ( 0 - 2pi )

    r = np.sqrt( x**2 + y**2 + z**2 )

    theta = arccos ( z / r )


    return theta, psi, phi, r


def predict(x, y, z, dx, dy, dt, cam, horizon):

        tstep = np.array([np.arange(0,horizon),]*len(x)).T

        # predicted position in cartesian
        new_x = x + dx * tstep/dt
        new_y = y + dy * tstep/dt
        new_z = z

        # cartesian to spherical
        zenith, azimuth, _, r = cart2sphere(new_x, new_y, new_z)

        # spherical to image coordinates
        nextX, nextY = cam.sphere2img(zenith, azimuth, rotate = True)

        return np.array(zenith), np.array(azimuth), np.array(nextX), np.array(nextY)


def arrowedLine(img, pt1, pt2, color,thickness=1,
    line_type=8, shift=0, tipLength=0.1):

    tipSize = np.sqrt((pt1[1]-pt2[1])**2+(pt1[1]-pt2[1])**2)*tipLength # Factor to normalize the size of the tip depending on the length of the arrow
    #if np.isnan(tipSize): return
    img = cv2.line(img, (pt1[0],pt1[1]), (pt2[0],pt2[1]),
                   color, thickness, line_type, shift)

    angle = np.arctan2( pt1[1] - pt2[1], pt1[0] - pt2[0] )
    p = [int(pt2[0] + tipSize * np.cos(angle + np.pi / 4)),
    int(pt2[1] + tipSize * np.sin(angle + np.pi / 4))]
    img = cv2.line(img, (p[0],p[1]), (pt2[0],pt2[1]),
                   color, thickness, line_type, shift)

    p[0] = int(pt2[0] + tipSize * np.cos(angle - np.pi / 4))
    p[1] = int(pt2[1] + tipSize * np.sin(angle - np.pi / 4))
    img = cv2.line(img, (p[0],p[1]), (pt2[0],pt2[1]),
                   color, thickness, line_type, shift)
