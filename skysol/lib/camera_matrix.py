import numpy as np
from numpy import log, sqrt, exp, tan, arctan2, sin, cos, arctan, degrees, radians, log, pi, arcsin, arccos
import time
import cv2
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import gc
from operator import itemgetter

class camera:

    def __init__(self,ini):

        self.scale = ini.scale

        # cameras imagesize in pixels
        self.imgsize = ini.imgsize

        # topocentric zenith angle
        self.theta = 0

        # pixels distance to image center
        #self.dtc = 0

        # radius of image
        self.radius = ini.radius

        # image center
        self.cx = int(ini.cy)
        self.cy = int(ini.cx)

        self.un_size = ini.grid_size

        # distance in pixels from image center to edge of fov (theta=90)
        self.fx = ini.fx
        self.fy = ini.fy

        # focal length
        self.f = 1.0

        # affine parameters
        self.c = ini.cam_c
        self.d = ini.cam_d
        self.e = ini.cam_e

        # polynomial coefficients from OCamCalib Calibration
        self.polyc = ini.polyc
        self.invpoly = ini.invpoly

        # used localization function / projection model
        self.pmodel = "taylor"

        # coordinate axis rotation angles for camera2world trafo
        self.rot_angles = ini.rot_angles

        # rotation matrix
        self.R = rotMat(self.rot_angles[0],self.rot_angles[1], \
                            self.rot_angles[2], rotZ=True)

        self.R2 = rotMat(self.rot_angles[0],self.rot_angles[1],-self.rot_angles[2], rotZ=True)
        self.R4 = rotMat(self.rot_angles[0],self.rot_angles[1], self.rot_angles[2], rotZ=False)

        # calculate the important image matrices
        self.__getImageMatrices(ini)







    def __getImageMatrices(self,ini):


        # Image coordinates
        imggrid = np.mgrid[0:self.imgsize[0],0:self.imgsize[1]]

        # centered absolute coordinates
        u = imggrid[0,:,:] - self.cx
        v = imggrid[1,:,:] - self.cy

        # centered normalized coordinates
        x0 = u / float(self.fx)
        y0 = v / float(self.fy)

        # distance to image center
        self.dtc_norm = np.sqrt( np.multiply(x0,x0) + \
                             np.multiply(y0,y0) )
        dtc = np.sqrt( np.multiply(u,u) + \
                             np.multiply(v,v) )

        del u,v

        x0[self.dtc_norm>1] = np.nan
        y0[self.dtc_norm>1] = np.nan


        # project image position to real world by applying the fisheye projection model
        z0, self.theta = self.projectionModel(x0,y0,dtc,model=self.pmodel,coeff=self.polyc)
        self.theta = np.float32(self.theta)

        # compute LUT for taylor model of the fisheye projection function
        if self.pmodel == "taylor": self.createPolyLut(dtc,self.polyc)

        del dtc

        # camera to world transformation
        x, y, z = camera2world(self.R2, x0, y0, z0)
        _, _, self.phi2, _ = cart2sphere( x, y, z )

        # Real world coordinates
        x0, y0, z0 = camera2world(self.R, x0, y0, z0)

        # get spherical coordinates
        _, psi, self.phi, r = cart2sphere( x0, y0, z0 )

        del x0, y0, z0







    def surfaceDist(self,cbh,saz,sza):
        """
        Calculates the distance (component-wise) of all pixels shadow to the cameras
        position (assumed they are cloudy with certain height cbh)

        :param cbh: cloud base height ( in meters )
        :type cbh: float

        :param sza: solar zenith angle ( radians )
        :type sza: float

        :param saz: solar azimuth angle ( radians )
        :type saz: float

        :returns x,y: distance in x and y direction ( meters )
        :type x,y: array with size of raw image
        """

        x = float(cbh) * (  tan(self.theta) * sin(self.phi) + tan(sza) * sin(saz) )
        y = -float(cbh) * (  tan(self.theta) * cos(self.phi) + tan(sza) * cos(saz) )

        return x, y





    def grid(self,image, map_x, map_y):
        """
        image to grid mapping
        """
        img = cv2.remap(image,map_x,map_y,cv2.INTER_NEAREST)

        return img


    def remap(self,image, map_x, map_y, interp="nearest"):
        """
        image to grid mapping
        """
        if interp == "nearest":
            img = cv2.remap(image,map_x,map_y,cv2.INTER_NEAREST)
        elif interp == "linear":
            img = cv2.remap(image,map_x,map_y,cv2.INTER_LINEAR)

        return img




    def undistort(self, image):
        """
        Camera2World Transformation for the whole image
        Return the transformed image, if the mapping matrices are given
        """
        image = cv2.remap(image,self.map_x,self.map_y,cv2.INTER_LINEAR)
        return image





    def projectionModel(self,x,y,r,model="stereographic",coeff=1.0):
        """
        apply the fisheye projection model which is a function of type
            theta = f(r) converting image radius to its
        radially symmetric true incidence angle

        coeff: the focal length
        """

        # camera projection model
        if model == "perspective":
            # perspective projection r = f * tan (theta)
            theta = arctan( r / ( coeff ) )
        elif model == "stereographic":
            # stereographic projection model r = 2 * f * tan( theta / 2 )
            theta = 2. * arctan( r / ( 2. * coeff ) )
        elif model == "equidistance":
            # equidistance projection model r = f * theta
            theta = r/ coeff
        elif model == "equisolid":
            # equisolid angle projection model r = 2 * f * sin(theta/2)
            theta = 2. * arcsin( r / ( 2. * coeff ) )
        elif model == "orthogonal":
            # orthodonal projection model r = f * sin( theta )
            theta = arcsin( r / float(coeff) )
        elif model == "linear":
            # linear scaling between 0 and pi/2
            theta = r * (pi/2.)
        elif model == "own":
            #self.theta = log( ( ( r + 1. ) / coeff )**(1/0.45) )
            theta = log( ( r + 1. )**(1/0.7) )
        elif model == "taylor":
            r = r / self.scale
            z = poly.polyval(r,coeff)
            z = self.scale * z / (-self.radius)
            r3d = np.sqrt( x**2 + y**2 + z**2 )
            theta = arccos ( z / r3d )
        else:
            raise "no or wrong projection model given"
        if model != "taylor":
            r3d = r / cos((pi/2.)-theta)
            z = np.multiply(r3d,cos(theta))
        del r3d
        return z, theta



    def createPolyLut(self,r,coeff):
        """
        creates a LUT for zenith angles according to the polynomial
        localization function

        Input:

        r = distance to image center matrix ( in pixel ) for each pixel
        coeff = coefficients of polynomial function
        """

        x = np.arange(self.fx)
        x1 = x / float(self.fx)
        dtc = sqrt(x**2 + x**2)
        z = -1 * poly.polyval(x,coeff)
        z = z / self.fx
        r = np.sqrt( x1**2 + z**2 )
        self.poly_lut = -1 * arccos ( z / r )


    def camera2image(self,x,y):

        x1 = np.round(self.fx * ( x * self.c + y * self.d ) + self.cx,0)
        y1 = np.round(self.fy * ( x * self.e + y ) + self.cy,0)
        return x1.astype(int), y1.astype(int)




    def sphere2cart(self,theta,phi,model="stereographic",filename="",coeff=1.0):
        """
        coordinate transformation from spherical to
        2d - cartesian coordinates by applying projection model

        Input: zenith angle ( theta ) and azimuth angle ( phi )
        in radians, projection model

        Output: x, y  cartesian coordinates
        """

        # camera projection model
        if model == "perspective":
            # perspective projection r = f * tan (theta)
            r =  coeff * tan(theta)
        elif model == "stereographic":
            # stereographic projection model r = 2 * f * tan( theta / 2 )
            r = 2. * coeff * tan( theta / 2. )
        elif model == "equidistance":
            # equidistance projection model r = f * theta
            r = coeff * theta
        elif model == "equisolid":
            # equisolid angle projection model r = 2 * f * sin(theta/2)
            r = 2 * coeff * sin( theta/2. )
        elif model == "orthogonal":
            # orthodonal projection model r = f * sin( theta )
            r = coeff * sin(theta)
        elif model == "linear":
            # linear scaling between 0 and pi/2
            r = theta / (pi/2.)
        elif model == "own":
        #r = coeff * exp(theta)**0.45 - 1
            r = coeff * exp(theta)**0.7 - 1
        elif model == "taylor":
            r = poly.polyval(theta,self.invpoly)
            r = r * self.scale
        else:
            raise argumentError

        x = r / float(self.fx) * sin ( phi )
        y = r / float(self.fy) * cos ( phi )

        return r,x,y


    def sphere2img(self,theta,phi,rotate=True):
        """
        Apply the forward camera model to single spherical coordinates
        like the sun position

        Input

        theta: incidence angle (radian)
        phi: azimuthal angle (radian) defined clockwise from north

        return x,y: image coordinates
        """

        # apply inverse fisheye projection
        dist, x, y = self.sphere2cart(np.array(-1*(np.pi/2.-theta)),phi,model=self.pmodel)

        # apply rotation matrix
        if rotate:
            x, y, z = camera2world(self.R, x, y, 1.0)
            return self.camera2image(-x, -y)
        else:
            x, y, z = camera2world(self.R2, x, y, 1.0)
            return self.camera2image(-x, -y)




    def tandiff(self,xsun,ysun,theta_sun):

        a = tan(self.theta[xsun,ysun])
        b = tan(theta_sun)
        return a-b


    def mapping_backward(self, x, y, x_res, y_res, map_size, max_angle, \
        cbh=1.,saz=0.,sza=0.,cbh_flag=True,shadow_flag=True ):

        """
        Transform pixel in 2D image space to 3D real world coordinate (in perspective projection)

        Fisheye projection is applied to determine pixels incidence angle.

        If cloud height is given the output will be given in meters.
        Otherwise the distance is normalized to the distance corresponding to the
        maximum viewing angle and converted in output image coordinates.

        For cloud shadow position computation solar position must be provided.

        Input:

        x: image coordinate in x direction
        y: image coordinate in y direction

        map_size: radius of final cloud map or grid in pixels/cells, used for scaling
        max_angle: maximum viewing angle ( radians ) for normalization (must be <90°)
        x_res: grid resolution in meters / pixel latitudinal
        y_res: grid resolution in meters / pixel longitudinal

        cbh: optional, cloud pixels base height
        saz: optional, solar azimuth angle ( radians )
        sza: optional, solar zenith angle ( radians )

        cbh_flag: optional, boolean flag
        shadow_flag: optional, boolean flag

        return: x,y position in either meters or pixels

        """


        x1 = []
        y1 = []
        if type(x) != np.ndarray:
            x = [x]; y = [y]

        # calculate displacement
        for i in range(0,len(x)):

            try:
                if cbh_flag:

                    # compute cloud position using cloud height and lens function
                    a = cbh * ( np.tan(self.theta[x[i],y[i]]) * np.sin(self.phi2[x[i],y[i]]) )
                    b = - cbh * ( np.tan(self.theta[x[i],y[i]]) * np.cos(self.phi2[x[i],y[i]]) )

                    if shadow_flag:

                        # calculate shadow position using cloud height and sun position
                        a = a - cbh * ( np.tan(sza) * np.sin(saz) )
                        b = b + cbh * ( np.tan(sza) * np.cos(saz) )


                    # convert to grid
                    a = a / x_res + map_size / 2.
                    b = b / y_res + map_size / 2.

                else:

                    # compute cloud position using lens function and normalize it to [0,1]
                    a = -np.tan(self.theta[x[i],y[i]]) * np.sin(self.phi2[x[i],y[i]]) / np.tan( max_angle )
                    b = np.tan(self.theta[x[i],y[i]]) * np.cos(self.phi2[x[i],y[i]])  / np.tan( max_angle )

                    # convert to grid
                    a = (-1. * a * map_size/2.) + map_size/2.
                    b = (-1. * b * map_size/2.) + map_size/2.

                x1.append(a); y1.append(b)

            except:

                x1.append(np.nan); y1.append(np.nan)


        return np.array(list(zip(x1,y1)))




    def forward(self, max_angle=80.):
        """
        apply forward camera model projecting according to the underlying
        fisheye projection. It determines the corresponding real world coordinates
        in the output image. As a result it creates the fisheye projection
        of a regular (rectilinear) grid. The function returns the mapping matrices
        for OpenCVs remapping function.
        """
        x = self.un_size/2 * np.tan(self.theta) * np.sin(self.phi2) / np.tan(np.radians(max_angle))
        y = self.un_size/2 * -np.tan(self.theta) * np.cos(self.phi2) / np.tan(np.radians(max_angle))

        map_inv_x = x + self.un_size/2
        map_inv_y = y + self.un_size/2

        return np.array(map_inv_x, dtype=np.float32), np.array(map_inv_y, dtype=np.float32)






    def backward(self,ini,cbh,saz,sza,limit=79.,cbh_flag=True,shadow_flag=True):
        """
        Create a mapping matrix (inverse model) or LUT that serves as input
        for OpenCVs remapping module. It includes the backward camera model for transformation
        of image pixels in 2D space to 3D real world coordinates. It is used for
        cloud (shadow) mapping to a given grid. The LUT has a variable size
        (usually grid size in meters divided by grid resolution).
        For output matrices map_x and map_y consists of the corresponding image pixel positions.
        """
        map_x = np.zeros(([self.un_size ,self.un_size ]), dtype=np.float32)
        map_y = np.zeros(([self.un_size ,self.un_size ]), dtype=np.float32)

        # define underlying meshgrid
        imggrid = np.mgrid[0:self.un_size ,0:self.un_size ]


        if cbh_flag:

            # convert to teuclidean distance from center of the grid in both directions
            # use initial grid definiions
            x = -ini.x_res * ( imggrid[0,:,:] - ini.grid_size/2. )
            y = -ini.y_res * ( imggrid[1,:,:] - ini.grid_size/2. )

            # cloud position
            a =   -1. * ( y / float(cbh) )
            b =   ( x / float(cbh) )

            # shadow displacement
            if shadow_flag:

                a = a + ( tan(sza)*sin(saz) )
                b = b + ( tan(sza)*cos(saz) )


            # Rotate coordinates with camera orientation
            a,b,z = camera2world(self.R4,a,b,0)

            phi0 = np.arctan2(a,b)
            phi = ( phi0 + 2.*np.pi ) % (2.*np.pi)
            phi[phi==0] = 0.00001

            theta = abs(np.arctan( np.divide( a , np.sin(phi) ) ))

            # mask horizon
            theta[np.degrees(theta)>limit] = np.nan

            # apply lens function
            r, _, _ = self.sphere2cart(-1.*(np.pi/2.-theta), \
                    0, model="taylor")


        else:


            # centered absolute coordinates
            u = -1. * ( imggrid[0,:,:] - ini.grid_size/2.) / ( ini.grid_size/2. )
            v = ( imggrid[1,:,:] - ini.grid_size/2.) / ( ini.grid_size/2. )

            # Rotate coordinates with camera orientation
            x, y, _ = camera2world(self.R4,v,u,0)

            # convert cartesian coordinates to polar coordinates
            phi0 = np.arctan2(x,y)
            phi = ( phi0 + 2.*np.pi ) % (2.*np.pi)
            phi[phi==0] = 0.00001

            # inverse mapping function
            theta = np.abs(np.arctan( np.divide( x * np.tan(np.radians(limit)), np.sin(phi) ) ) )

            # mask horizon
            theta[theta> np.radians(limit)] = np.nan

            # apply lens function to retrieve radius according to incidence angle
            r, _, _ = self.sphere2cart(-1.*(np.pi/2.-theta), 0, model="taylor")


        r[r>self.radius]=np.nan
        i =  r * np.sin(phi)
        j =  r * -np.cos(phi)

        map_x =  np.asarray(i  + self.cy,dtype=np.float32)
        map_y =  np.asarray(j  + self.cx,dtype=np.float32)

        return map_x, map_y





    def backward_old(self,ini):

        #####################################################
        # map_x, map_y is LUT for the undistorted image where
        # the corresponding pixel in the raw image is
        #####################################################

        rows = self.un_size
        cols = self.un_size

        # define mapping matrices ( size is given by config file )
        map_x = np.zeros(([self.un_size ,self.un_size ]), dtype=np.int32)
        map_y = np.zeros(([self.un_size ,self.un_size ]), dtype=np.int32)


        # init matrices
        imggrid = np.mgrid[0:self.un_size ,0:self.un_size ]
        x = ( imggrid[0,:,:] - cols/2. ) / (cols/2.)
        y = ( imggrid[1,:,:] - rows/2. ) / (rows/2.)
        x[np.sqrt(np.multiply(x,x)+np.multiply(y,y))>1]=np.nan
        y[np.sqrt(np.multiply(x,x)+np.multiply(y,y))>1]=np.nan


        # rotate coordinates
        x,y,z = camera2world(self.R,x,y,0)

        # incidence angle with already scaling to horizon
        theta = np.arctan(np.sqrt(np.multiply(x,x)+np.multiply(y,y)) * np.tan(radians(ini.horizon)) )

        # azimuth angle ( orientated north )
        phi = arctan2(-1.*y,x) + pi/2.
        phi = ( phi + 2.*pi ) % (2.*pi)

        # get distance to center for each pixel
        r, _, _ = self.sphere2cart(-1.*(np.pi/2.-theta), \
                0, model="taylor")

        # calculate relative coordinates
        r[r>self.radius]=np.nan
        i =  r * np.sin(phi)
        j =  r * -np.cos(phi)

        # calculate abolsute image coordinates
        map_x =  i  + self.cy
        map_y =  j  + self.cx

        map_x = np.asarray(map_x,dtype=np.float32)
        map_y = np.asarray(map_y,dtype=np.float32)

        # clean up
        return map_x, map_y


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


    # azimuth counterclockwise
    psi = arctan2(y,x)

    # azimuth clockwise counting from top of image
    phi = arctan2(-1*y,x) + pi/2.
    phi = ( phi + 2.*pi ) % (2.*pi) # ( 0 - 2pi )

    r = np.sqrt( x**2 + y**2 + z**2 )

    theta = arccos ( z / r )


    return theta, psi, phi, r




def world2camera(R,x,y,z):
    """
    calculates the world2camera transformation.
    It applies the inverse of the rotation matrix
    to the coordinate-triple ( x,y,z )
    """

    R = np.linalg.inv(R)

    x1 = np.multiply( R[0,0],x ) + \
        np.multiply( R[0,1],y ) + \
        np.multiply( R[0,2],z )
    y1 = np.multiply( R[1,0],x ) + \
        np.multiply( R[1,1],y ) + \
        np.multiply( R[1,2],z )
    z1 = np.multiply( R[2,0],x ) + \
    np.multiply( R[2,1],y ) + \
    np.multiply( R[2,2],z )

    return x1,y1,z1


def camera2world(R,x,y,z):
    """
    calculates the camera2world transformation.
    It applies the rotation matrix
    to the coordinate-triple ( x,y,z )

    Input:

    rotation matrix (R)
    object coordinates x,y,z

    """
    x1 = np.multiply( R[0,0],x ) + \
        np.multiply( R[0,1],y ) + \
        np.multiply( R[0,2],z )
    y1 = np.multiply( R[1,0],x ) + \
        np.multiply( R[1,1],y ) + \
        np.multiply( R[1,2],z )
    z1 = np.multiply( R[2,0],x ) + \
        np.multiply( R[2,1],y ) + \
        np.multiply( R[2,2],z )

    return x1, y1, z1


def rotMat(alpha,beta,gamma,rotZ=True):
    """
    defines the rotation matrix.

    with the three rotation angles ( for each axes )
    single rotation matrixes are defined an later combined to one.

    Attention: a rotation by -90 degrees ( z-axes ) is applied
    before to rotate the coordinate system from the image system
    ( x: top-down, y: left-right) to hemispheric system ( x: left-right,
    y: down-top ).

    Input:

    three rotation angles for each axes ( alpha, beta, gamma ) in radians
    """

    # rotate z-axes before
    if rotZ: gamma = gamma - pi/2.

    # default rotatin matrices for single axis
    Rx = np.matrix( ( (1,0,0),                    (0, cos(alpha),-sin(alpha)), (0, sin(alpha), cos(alpha)) ) )
    Ry = np.matrix( ( (cos(beta),0,sin(beta)),    (0, 1, 0),                   (-sin(beta), 0, cos(beta))  ) )
    Rz = np.matrix( ( (cos(gamma),-sin(gamma),0), (sin(gamma),cos(gamma), 0),  (0, 0, 1)                   ) )


    Rot = np.mat(Rx) * np.mat(Ry) * np.mat(Rz) #* np.mat(rotz)

    return Rot
