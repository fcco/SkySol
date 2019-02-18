import cv2
import numpy as np


def equi_distance(img, cam, lc=[255,255,0], lw=5, add_text=False):
    """
    Draw (in perspective projection) equidistant circles
    Project radial equidistant lines to fisheye space
    """
    for h in 1.0, 2.0, 3.0, 4.0, 5.0:

        # h is unit distance from camera center
        theta = np.arctan(h)
        # forward camera model
        r,_,_ = cam.sphere2cart(-1*(np.pi/2-theta), 0, model="taylor")
        cv2.circle(img, (cam.cy,cam.cx), int(r), lc, thickness=lw, lineType=4)
        if add_text: cv2.putText(img, str(h), (cam.cx-15,cam.cy-int(r)+30), \
                         cv2.FONT_HERSHEY_PLAIN, 3, color=lc, \
                         thickness=lw, lineType=8)


def draw_grid(img, cam):
    """
    Draw orthogonal grid lines with equal distance and orientated north to input
    fisheye image.

    It project a regular grid to fisheye space using the forward camera model.
    """

    # define grid
    x = np.zeros_like(img)
    x[::100,:] = 1
    x[:,::100] = 1

    map_x, map_y = cam.forward(xsize=img.shape[0], ysize=img.shape[1], height=1, max_angle=85.)

    mask = cam.grid(img,map_x,map_y)
    img[mask] = 0

    return img

def draw_boundaries(img, bounds):
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_float

    return mark_boundaries(img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), bounds)


def incidence_angle(img, cam, lc=[0,0,255], lw=5, add_text=False):
    """
    Draw circles of equal incidence angles
    """
    for h in 10,20,30,40,50,60,70,80,89:
        # Forward camera model
        r,_,_ = cam.sphere2cart(-1*np.radians(90-h), 0, model="taylor")
        cv2.circle(img,(cam.cy,cam.cx), int(r), lc, thickness=lw, lineType=8)
        if add_text: cv2.putText(img, str(h), (cam.cx-15,cam.cy-int(r)+30), \
                                 cv2.FONT_HERSHEY_PLAIN, 3, color=lc, \
                                 thickness=lw, lineType=8)



def circumsolar_area(img,theta,phi,sza,saz, lc=[0,255,0]):
    """ Computes angular distance from sun to each pixel and draws isolines
    """
    elev = np.pi/2 - theta
    sea = np.pi/2 - sza
    # Angular distance  (Sun-Pixel-Angle SPA) to the sun for each pixel
    # (Great circle distance)
    spa = np.arccos(np.sin(sea)*np.sin(elev) + np.cos(sea) * \
        np.cos(elev) * np.cos(phi-saz) )

    for n in 0.5,3,5,10,15,20:
        pix_sun = (spa >= np.radians(n-0.15)) & (spa <= np.radians(n+0.15))
        if img.ndim == 3:
            img[pix_sun,:] = lc
        elif img.ndim == 1:
            img[pix_sun] = 255


#def sector(img,phi,cloud_direction,angle=np.radians(30)):
#
#    pix_sector = (np.round((phi - angle),1) == np.round(cloud_direction,1))
#    img[pix_sector,:] = [0,0,0]
#    pix_sector = (np.round((phi + angle),1) == np.round(cloud_direction,1))
#    img[pix_sector,:] = [0,0,0]


def forecast_path(img, x, y, lcolor=[50,50,50]):
    """ Draw a line along the forecast path"""
    for i in range(61,len(x),61):
        if (x[i-1] <= 0) or (x[i] <= 0): continue
        #cv2.arrowedLine(img,(y[i],x[i]),(y[i-50],x[i-50]), lcolor, 15, tipLength=0.2)
        cv2.arrowedLine(img,(y[i],x[i]),(y[i-50],x[i-50]), [25,200,250], 13, tipLength=0.5)


def cloud_path(img, y, x, lcolor=[150,0,0]):

     """ Draw a line for each cloud motion path"""
     for j in range(0,x.shape[1], 50):
        for i in range(50,x.shape[0]-10, 60):
            if (x[i-50,j] <= 0) or (x[i,j] <= 0): continue
            cv2.arrowedLine(img,(y[i-50,j],x[i-50,j]),(y[i,j],x[i,j]), lcolor, 5, tipLength=0.5)


def sunpath(img, dt, ini, cam):
    """
    Draws any line through given image, but was intenionally build for
    drawing the path of the sun during a day
    """
    from skysol.lib import solar
    from datetime import timedelta

    sunpos = solar.compute( [dt] , ini.lat0, ini.lon0, method="spencer")

    # Detect sun position
    #img, circles = detect_houghcircles(img, cam.theta, cam.phi2, sunpos['zenith'], sunpos['azimuth'])

    base = dt.replace(hour=0,minute=0,second=0)
    dt = [ base + timedelta(seconds=i*600) for i in range(0,144)]
    sun = solar.compute( dt , ini.lat0, ini.lon0, method="spencer" )

    x, y = cam.sphere2img(sun['zenith'], sun['azimuth'], rotate=True)
    ind = (x > 0) & ( y > 0 )
    x = x[ind]
    y = y[ind]

    # Draw sun path
    for i in range(1,len(x)):
        cv2.line(img,(y[i-1],x[i-1]),(y[i],x[i]), [60,60,60], 8, lineType=4)

    x, y = cam.sphere2img(sunpos['zenith'], sunpos['azimuth'], rotate=True)

    # Draw sun position
    img = cv2.circle(img, (y,x), 10, (255,0,0), 5)

    return img


def detect_houghcircles(img,theta,phi,sza,saz):
    import matplotlib.pyplot as plt
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,0]

    hue = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,0]

    elev = np.pi/2 - theta
    sea = np.pi/2 - sza
    # Angular distance  (Sun-Pixel-Angle SPA) to the sun for each pixel
    # (Great circle distance)
    spa = np.arccos(np.sin(sea)*np.sin(elev) + np.cos(sea) * \
        np.cos(elev) * np.cos(phi-saz) )


    # resize image
   # pix = (spa <= np.radians(15)) & ( hue == 0 )

    ind = np.where( (spa <= np.radians(15)) & ( hue == 0 ))

    if len(ind) == 0: return img, None

    #xmin = np.min(np.where(pix)[0])
    #xmax = np.max(np.where(pix)[0])
    #ymin = np.min(np.where(pix)[1])
    #ymax = np.max(np.where(pix)[1])

    #gray_img_small = gray_img[xmin:xmax,ymin:ymax]
    #gray_img_small = cv2.medianBlur(gray_img_small,11)


    ##detect circles
    #circles = cv2.HoughCircles(gray_img_small,cv2.HOUGH_GRADIENT,1,100,
    #                        param1=30,param2=50,minRadius=50,maxRadius=250)

    #sunxc, sunyc = (xmax-xmin)/2 + xmin, (ymax-ymin)/2 + ymin
    sunxc, sunyc = np.mean(ind[0]), np.mean(ind[1])

    try:
        cv2.circle(img,(int(sunyc),int(sunxc)),10,(51,100,0),10)
    except:
        pass

    #if circles is None: return img, None
    #circles = np.uint(np.around(circles))
    #cnt = 0
    #for i in circles[0,:]:
    #    # draw the outer circle
    #    cv2.circle(img,(int(i[0] + ymin),int(i[1] + xmin)),i[2],(0,255,0),2)
    #    # draw the center of the circle
    #    cv2.circle(img,(int(i[0] + ymin),int(i[1] + xmin)),2,(0,0,255),3)
    #    circles[0,cnt,0] = i[0] + xmin
    #    circles[0,cnt,1] = i[1] + ymin
     #   cnt =+ 1



    return img, (sunxc, sunyc)
