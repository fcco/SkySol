import csv
import cv2
import gc
import numpy as np

def cloud_cover(cmap):

    # Number of masked pixels
    nmaskpix = np.sum(cmap == 0)

    # Total number of non-masked pixels ( sky pixels + cloud pixels )
    nskypix = np.sum(cmap > 0)

    # Number of cloudy pixels
    ncloudpix = np.sum(cmap == 100)

    # Number of clear sky pixels
    ncskpix = np.sum(cmap == 1)

    # Cloud Coverage
    cc_cmap = 100. * ( 1 -( ncskpix / float(nskypix)) ) #100.*float(ncloudpix)/float(nskypix+ncloudpix)

    return cc_cmap



def getSunPixels(sun_zenith,sun_azimuth,sun_angle,cam):
    """
    The function returns a matrix of the image
    pixels "around" the sun. With sun position given
    as sun zenith angle and azimuth angle and the
    sun solid angle given, it calculates the pixels
    in the image in the corresponding sun circle. A
    matrix is given with elements outside the region
    of interest as NaN

    Input:

    sun_zenith: Sun zenith angle in radians
    sun_azimuth: Sun azimuth angle ( 0-2pi in radians
    form north clockwise )
    sun_angle: sun solid angle in radians

    Output:

    Vector with image pixel indizes according to sun area
    """

    # define maximum and minimum of azimuth and zenith angle
    szmax = sun_zenith + sun_angle
    szmin = sun_zenith - sun_angle
    samax = sun_azimuth + sun_angle
    samin = sun_azimuth - sun_angle


def sun_area(img,cam,sza,saz,sunarea=3):

    hsv = cv2.cvtColor(img.orig_color,cv2.COLOR_BGR2HSV)

    # Angular distance to the sun for each pixel
    dist2sun = np.sqrt(np.square(cam.theta - sza) + \
        np.multiply(np.square(np.sin(sza)) , np.square(cam.phi2 - saz) ) )

    # number of pixels inside sunarea
    pix_sun = (dist2sun <= np.radians(sunarea))

    # pixels which are in sun area and saturated
    con = (pix_sun & (hsv[:,:,0] == 0))

    # Percentage of saturated pixels in circumsolar region
    pct_circum = np.sum(con) / float(np.sum(pix_sun))

    return pct_circum


def gray_coeff(img, cam, dist2sun):

    gamma = np.cos(cam.theta)
    gray = img.copy()
    gray[img==0] = np.nan
    gray[img==1] = 0
    gray = gray / 255.

    gamma[gray==0] = 0.0
    # mask sunspot
    gamma[dist2sun <= np.radians(3)] = np.nan

    f1 = np.nansum(np.multiply(gamma,gray))
    f2 = np.nansum(gamma)

    return np.divide(f1,f2) #, np.nanmean(gray)
