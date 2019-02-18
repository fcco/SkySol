import cv2
import sys
from datetime import datetime, timedelta
import os
import time
import pytz
import numpy as np

from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import matplotlib.pyplot as plt



class image:

    def __init__(self, ini, imglist, imgnum, cap="", filename=""):

        self.orig_color = 0             # Original image in true color
        self.orig_gray = 0              # Original image in gray color
        self.binary_color = 0           # Binary Cloud/Sky image with 3 channels
        self.cloud_bool = 0             # Binary Cloud/Sky image with boolean True for cloud
        self.sky_bool = 0               # Binary Cloud/Sky image with boolean True for sky
        self.binary_gray = 0            # Binary Cloud/Sky image with single channel
        self.rbr = 0                    # Red-Blue-Ratio Image with single channel
        self.datetimestr = ''
        datetime_old = ini.actdate
        self.flipping = ini.flipping    # flip image - 0: horizontal, 1: vertical, -1: both


        # image's timestamp
        self.actdate = 0

        # exit code
        self.exit = 1

        # get image filename
        if filename == "" and cap == "":
            if imgnum == -999:
                self.filename = imglist[-1]
            else:
                try:
                    self.filename = imglist[imgnum]
                except IndexError:
                    sys.exit("All images completed! -> Exit!")
        elif filename != "":
            self.filename = filename
            if imgnum < 0:
                actdate = datetime.strptime(self.filename.split(os.sep)[-1][0:15], ini.datetimefmt)
                oldstr = self.filename.split(os.sep)[-1][0:15]
                newdate = actdate + timedelta(seconds=+ini.camera_res)
                newstr = newdate.strftime(ini.datetimefmt)
                self.filename = self.filename.replace(oldstr,newstr)
        elif cap != "":
            self.filename = ""




        # Take current time for live mode
        if ini.live:
            if cap != "":
                self.locdate = pytz.timezone(ini.tz).localize(datetime.utcnow(),is_dst=False)
            else:
                self.locdate = datetime.strptime(self.filename.split(os.sep)[-1][0:15], ini.datetimefmt)
                self.locdate = pytz.timezone(ini.tz).localize(self.locdate,is_dst=False)
            self.actdate =  self.locdate.astimezone(pytz.UTC)
            self.datetimestr = self.actdate.strftime(ini.datetimefmt)
        else:
            self.locdate = datetime.strptime(self.filename.split(os.sep)[-1][0:15], ini.datetimefmt)
            self.locdate = pytz.timezone(ini.tz).localize(self.locdate,is_dst=False)
            self.actdate =  self.locdate.astimezone(pytz.UTC)
            self.datetimestr = self.actdate.strftime(ini.datetimefmt)

        # Update Settings
        ini.datestr = self.actdate.strftime("%Y%m%d")

        # Time between two subsequent images
        if imgnum == 0:
            self.timedt = 0
        else:
            self.timedt = int((self.actdate - datetime_old.replace(tzinfo=pytz.timezone(ini.tz))).total_seconds())

        # Wait until time between last image and now is at least x seconds
        if ini.live:
            shift = (datetime.utcnow().replace(tzinfo=pytz.utc) - self.actdate).seconds
            if shift < 60: time.sleep(60-shift)
            #if shift < 10: time.sleep(10-shift)

        # load the image
        self.load_image(ini, self.filename, cap=cap)

        # if loading was successful
        if self.exit:
            self.orig_color = flip_image(self.orig_color, self.flipping)
            self.orig_color = resize(self.orig_color, ini.scale)
            self.orig_gray = cv2.cvtColor(self.orig_color, cv2.COLOR_BGR2GRAY)
            self.shape = self.orig_gray.shape
            ini.actdate = self.actdate

        # Boolean to indicate if the image is useful for inspecting
        self.useful_image = True



    def load_image(self, ini, filename, cap="", quiet=True):
        # Live Mode
        if ini.live:
            # Get images from livestream
            if cap != "":
                ret, self.orig_color = cap.read()
                if ~ret:
                    print("Could not read from livestream. Check connection to webcam, proxies, etc... or switch to filemode -> ini.live_stream = False")
                    sys.exit()
            else:
                # get images from filesystem
                cnt = 0
                print("Waiting for image " + filename)

                while cnt < 600:
                    if os.path.isfile(filename):
                        try:
                            # Open image (wait 1s because sometimes image is not written completely)
                            time.sleep(1)
                            self.orig_color = cv2.imread(filename)
                            if self.orig_color.dtype == np.dtype('uint8'):
                                break
                            else:
                                continue
                        except IOError:
                            self.exit = 0
                    else:
                        time.sleep(2)
                        cnt += 2
                        continue


        # Archive Mode
        else:
            if os.path.isfile(filename):
                try:
                    self.orig_color = cv2.imread(filename)
                    if type(self.orig_color) is not np.ndarray: self.exit = 0
                except IOError:
                    self.exit = 0


            else:
                self.exit = 0



    def check_quality(self,mask):

        img = np.array(self.orig_gray, dtype=np.float)
        img[mask] = np.nan
        # too bright ( overexposed )?
        if ( np.nanmean(img) > 210 ):
            return False
        else:
            return True




    def compute_rbr(self, img, mask):
        """
        Computes the Red to Blue ratio
        and returns a float32 array with RBR Values

        Masked ares are set to 0
        RBR > 5 are set to nan
        """
        # Compute Red to Blue ratio
        img = np.array(img, dtype=np.float32)
        rbr = np.divide(img[:, :, 2], img[:, :, 0])
        rbr[rbr > 5] = np.nan
        rbr[mask] = 0

        return rbr

    def segment_clouds(self, threshold):

        # RBR-CSL thresholding method
        self.sky_bool = self.rbr <= threshold
        self.cloud_bool = self.rbr > threshold
        self.cloud_trans = (self.rbr <= threshold + 0.02) & (self.rbr > threshold)
        self.sky_trans = (self.rbr <= threshold) & (self.rbr > threshold-0.02)

    def rbr_csl_correction(self, ini, cam, csl, tstamp, saz, sza, mask):

        csl_ts = csl.read_csl(tstamp, saz, sza)
        if csl_ts >= 0:

            tzi = pytz.timezone(ini.tz)
            csl_dt = tzi.fromutc(datetime.utcfromtimestamp(csl_ts))
            idir = datetime.strftime(csl_dt,"%Y%m%d")
            cslfile = datetime.strftime(csl_dt, ini.picpath + os.sep + idir + os.sep + ini.datetimefmt + '.' + ini.imgformat)
            if not os.path.exists(cslfile):
                print("Clear sky reference image %s cannot be found. Check image origin!" % cslfile)
                return 0
            # Read image
            img = cv2.imread(cslfile)
            print("Clear sky reference read from ", cslfile)
            img = resize(img, ini.scale)
            # flip image
            self.cslimage = flip_image(img, self.flipping)
            # compute RBR
            self.cslimage = self.compute_rbr(self.cslimage, mask)
            # median blurring to reduce noise (transform to uint8 in order to use medianBlur)
            self.cslimage = np.float32(cv2.medianBlur(np.uint8(100 * self.cslimage), ini.median_filter)) / 100

            # compute sun-pixel-angle image
            elev = np.pi/2 - cam.theta
            sea = np.pi/2 - sza
            spa = np.arccos(np.sin(sea)*np.sin(elev) + np.cos(sea) * \
                np.cos(elev) * np.cos(cam.phi2-saz) )

            # Correction due to brightness in circumsolar area ( grade of saturation )
            radius = ini.sunspot
            ind = spa < np.radians(radius)
            # sun intensity factor
            cslfac = ini.csl_weighting * np.mean(self.orig_gray[ind]) / 255.

            # Correction due to each pixels intensity
            grayi = np.asarray(self.orig_gray,dtype=np.float32)

            # Correction function
            self.rbr = self.rbr_orig - \
                self.cslimage * cslfac -  self.rbr_orig**1.5 * ini.clear_correction * (grayi - 210)
            # Method as described in thesis
            #self.rbr = self.rbr_orig - \
            #    self.cslimage * (cslfac - ini.clear_correction * (grayi - 210 ))
            self.rbr[mask] = np.NaN

            return 1 # CSL Libary adjustment

        else:

            print("No clear sky reference found")
            return 0 # No correction





    def make_binary(self, mask):
        """
        Define new arrays based on segmented sky images

        binary_gray - segmented with grayscale colors
        binary_color - segmented with RGB colors for visualization

        """
        # binary image ( used for shadows )
        self.binary_gray = np.asarray(self.orig_gray, dtype=np.float32)
        self.binary_gray[self.sky_bool] = 1                       # csky
        self.binary_gray[self.cloud_bool] = 100                   # cloud
        self.binary_gray[mask] = 0      # mask

        # RGB image for visualization purposes
        self.binary_color = self.orig_color.copy()
        self.binary_color[self.sky_bool, :] = [ 255, 0 , 0 ]         # sky
        self.binary_color[self.cloud_bool, :] = [ 255, 255 , 255 ]   # cloud
        self.binary_color[mask, :] = 0     # mask

        # Mark transient regions
        #self.binary_color[self.sky_trans, :] = [255,255,0]
        #self.binary_color[self.cloud_trans, :] = [160,160,160]

        del self.sky_bool #,self.cloud_bool

        return self.binary_gray.astype(np.float32)


    def median_filter(self, kernel, mask):

        if kernel is not None:
            self.binary_color = cv2.medianBlur(self.binary_color,kernel)
            self.binary_gray = cv2.medianBlur(np.uint8(self.binary_gray),kernel)
            self.binary_gray[mask] = 0 # mask
            self.binary_color[mask,:] = 0 # mask

    def filter(self, img, method="meanShift"):

        if method == "meanShift":
            # Smoothens the image
            img = cv2.pyrMeanShiftFiltering(img, 17,17)

        return img


    def label_clouds(self, img):

        """
        Label connected regions of an integer array.
        """
        labels = measure.label(img, neighbors=None, background=None, return_num=False, connectivity=None)
        clouds = measure.regionprops(labels, intensity_image = self.rbr)

        return labels, clouds


    def watershed(self, img):

        """ watershed a segmented image """
        D = ndimage.distance_transform_edt(img)
        localMax = peak_local_max(D, indices=False, min_distance=30,
			labels=img)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=img)
        clouds = measure.regionprops(labels, intensity_image = self.rbr)

        return labels, clouds


    def mark_boundaries(self, img, labels):

        # Marking the contours of every cloud
        img = mark_boundaries(img, labels, color = (0,1,1), outline_color = (1,1,0), background_label= 0)

        return img





    def artifical_object_detection(self, img_old, qflag, img_clr, resize=None):
        """
        Method to detect artifical objects like birds or human beiings that blocking
        the sky view

        It uses the concept of superpixel segmentation to derive groups of pixels
        that share color, texture, ... depending on the segmentation algorithm used.

        In a next step, empirical derived thresholds and tests are performed to decide
        if a certain segment is not sky or cloud.

        The parts are masked and the masked is returned
        """

        # The dimension of the resized image the algorithm should work on
        if resize is None:
            resize = self.orig_gray.shape

        # Load mask
        mask = np.zeros_like(self.orig_gray)
        mask[self.mask_bool] = 1
        mask[~self.mask_bool] = 0
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask, dtype=np.bool)

        # Resizing images
        img = cv2.resize(self.orig_color, resize, interpolation=cv2.INTER_NEAREST)
        img_gray = cv2.resize(self.orig_gray, resize, interpolation=cv2.INTER_NEAREST)
        img_old = cv2.resize(img_old, resize, interpolation=cv2.INTER_NEAREST)
        img_lst_clr = cv2.resize(img_clr, resize, interpolation=cv2.INTER_NEAREST)

        # Masking
        img[mask] = 0
        img_gray[mask] = 0
        img_old[mask] = 0
        img_lst_clr[mask] = 0

        # Segmentation
        segments = np.array(segmentation(img, method="felzenswalb"))

        # The number of segments found
        self.number_of_segments = len(np.unique(segments))

        # Statistics for single segments/superpixels
        regions = measure.regionprops(segments, intensity_image = img_gray)

        # Initializing variables used in the loop
        area_bird = 0

        mean_intensity = np.nanmean(img_gray[~mask])

        new_mask = np.zeros_like(img_gray, dtype=np.bool)

        # new part
        stats = np.array([ (SPix.area, SPix.label, SPix.mean_intensity, SPix.extent) for SPix in regions ],
                         dtype=[('area', 'i4'),('label', 'i4'), ('mean_intensity', 'f4'), ('extent','f4')])

        # average RGB of latest clear sky image
        rgb_clr = np.array([ np.nanmean(img_lst_clr[segments==label],axis=(0)) for label in stats['label'] ])

        # average RGB of current image
        rgb_img = np.array([ np.nanmean(img[segments==label],axis=(0)) for label in stats['label'] ])

        blueval0 = np.abs(rgb_clr[:,2] - (rgb_clr[:,0] + rgb_clr[:,1]) / 2)
        blueval1 = np.abs(rgb_img[:,2] - (rgb_img[:,0] + rgb_img[:,1]) / 2)

        int0 = np.nanmean(rgb_clr,axis=1)
        int1 = np.nanmean(rgb_img,axis=1)

        # Bird conditions
        size_limit = 0.02 * img.shape[0]**2
        int_con = ( np.abs(int0 - int1) > (0.2 * int0) ) & ( stats['area'] > size_limit )
        blue_con = ( np.abs(blueval0 - blueval1) > (0.2 * blueval0) ) & (stats['area'] > size_limit)

        bird_candid = int_con | blue_con

        pix_mask = np.zeros_like(segments, dtype=np.bool)
        for lab in stats['label'][bird_candid]:
            pix_mask[segments==lab] = True
        area_bird = np.sum(pix_mask)
        try:
            new_mask[pix_mask] = True
            img_lst_clr[~new_mask] = img[~new_mask].copy()
        except:
            pass

        # Droplet conditions
        size_limit = 0.0002 * img.shape[0]**2
        drop_candid = (stats['area'] > size_limit) & \
            (stats['area'] < 5000) & \
            (stats['extent'] > 0.25) & \
            (stats['mean_intensity'] < (0.7 * np.nanmean(img[img>0])))

        pix_mask = np.zeros_like(segments, dtype=np.bool)
        for lab in stats['label'][drop_candid]:
            pix_mask[segments==lab] = True
        area_drop = np.sum(pix_mask)
        try:
            new_mask[pix_mask] = True
        except:
            pass

        height, width = img.shape[:2]

        if area_bird >= (height * width - (len(img[~mask]))) * 0.3:
            self.useful_image = False


        new_mask = np.array(new_mask, dtype=np.uint8)
        new_mask = cv2.resize(new_mask, self.mask_bool.shape)
        self.new_mask = np.array(new_mask, dtype=np.bool)

        self.segments = segments

        self.area_bird = area_bird / len(img_gray[~mask])
        self.area_drop = area_drop / len(img_gray[~mask])

        return self.new_mask, qflag, img_lst_clr



def get_mask(ini,img,dist):
    """
    Read the user defined static image mask
    and return a boolean array with masked parts beiing true
    """
    tmp = cv2.imread(ini.maskfile)
    tmp = flip_image(tmp, ini.flipping)
    if type(tmp) != np.ndarray:
        print("No image mask %s found! Continue without static mask" % ini.maskfile)
        tmp = np.zeros_like(img,dtype=np.uint8)
        tmp = resize(tmp, ini.scale)
    else:
        # Black color = True; white color = False
        tmp = resize(tmp, ini.scale)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY) < 100
    mask_bool = tmp != 0
    mask_bool[dist>1] = True

    tmp = 0; del tmp

    return mask_bool



def droplets():

	image_org = cv2.imread(img)
	output = image_org.copy()
	image_gray_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
	image_gray = resize(image_gray_org)

	# initializing the mask
	mask_org = cv2.imread('mask.png') #Load mask
	mask = resize(mask_org) # Used to compute area
	mask_gray= cv2.cvtColor(mask_org, cv2.COLOR_BGR2GRAY) #Create gray scale mask

	image_org[mask_org==0] = 255	#Applying mask

	image = resize(image_org)		#resizing image

	segments = felzenszwalb(img_as_float(image), scale=177 , sigma=1.2, min_size=208)
	#segments = slic(img_as_float(image), n_segments = 100, sigma = 5)

	# Showing the output of segmentation
	fig = plt.figure("Superpixels")
	ax = fig.add_subplot(1, 1, 1)
	ima = mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)
	ax.imshow(ima)
	plt.axis("off")
	plt.show()

	io.imsave("C:\\Users\\Hazem\\Documents\\Sky\\Segmentation\\Test\\Segmented\\sgm(177,1.2,208)th(0.7)(0.5)" + str(cnt) + ".jpg", ima)

	regions = measure.regionprops(segments, intensity_image = image_gray)

	mask2 = np.zeros(image.shape[:2], dtype = "uint8")


	for SPix in regions:

		if SPix.area > 100 and SPix.area < 5000 and SPix.extent > 0.25 and SPix.mean_intensity < 0.7 * np.mean(image):
			if (abs(SPix.centroid[0] - 300) > 240 or abs(SPix.centroid[1] - 300) > 240):
				mask2 [segments == SPix.label] = 255
				flag = False

			else:
				pixmask = (segments==SPix.label) & (image_gray < (0.5 * np.mean(image)))
				mask2[pixmask] = 255


	image[mask2<255] = 0
	image[mask2==255] = 255

	io.imsave("C:\\Users\\Hazem\\Documents\\Sky\\Segmentation\\Test\\Segmented\\msk(177,1.2,208)th(0.7)(0.5)" + str(cnt) + ".jpg", image)

	cv2.imshow("Masked", image)
	cv2.waitKey(0)

	print ("Possible Droplets: ", flag)



def segmentation(img, method="slic"):

    if method == "felzenswalb":
        #segments = felzenszwalb(img, scale=850, sigma=1.0, min_size=1000)
        #segments = felzenszwalb(img, scale=180, sigma=1.2, min_size=200)
        segments = felzenszwalb(img, scale=50, sigma=1.5, min_size=150)
    elif method == "slic":
        segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
    elif method == "quick":
        segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

    return segments



def sun_frac(img,sza,saz,theta,azimuth,distance=2):
    """
    Calculates the fraction of saturated pixels inside the sunspot region

    Parameters:
    -----------
    img: array-like
        1d-channel image (gray-level)
    sza: float
        solar zenith angle in radians
    saz: float
        solar azimuth angle in radians
    theta: array_like
        array (image-size) of zenith angles for each pixels
    azimuth: array-like
        array (image-size) of azimuth angles for each pixels
    distance: float
        angular distance (radians) which defines the sunspot area
    Returns:
    --------
    sf: float
        ratio of pixels in the given sunspot area which are saturated

    """
    import matplotlib.pyplot as plt
    distance = 10
    distarr = np.sqrt( (theta - sza)**2 + np.sin(sza)**2 * (azimuth - saz )**2 )
    r = (distarr <= np.radians(distance))
    sunarea = np.count_nonzero(r)

    img[r] = 0
    plt.clf()
    plt.imshow(img)
    plt.draw()
    plt.savefig('/home/thomas.schmidt/test.png')
    bright = np.count_nonzero(img[r] >= 150)

    sf = bright / float(sunarea)

    return sf


def flip_image(frame, flip):
    frame = cv2.flip(frame, flip, frame)
    return frame

def resize(frame, scale=1):
    height, width = frame.shape[:2]
    new_shape = (int(height * scale), int(width * scale))
    return cv2.resize(frame, new_shape)

def filter(ini, sdate, images, edate="", dtfmt="%Y%m%d_%H%M%S"):
    """
    Return only those image filenames from the given input image list,
    that fall between start and end date defined by input arguments
    -s YYYYMMDD_HHMMSS -e YYYYMMDD_HHMMSS or after input date given by
    -d YYYYMMDD
    """
    new_list = []
    if edate == "": edate = datetime(2050, 1, 1, 0, 0, 0)
    sdate = pytz.UTC.localize(sdate,is_dst=False).astimezone(pytz.UTC)
    edate = pytz.UTC.localize(edate,is_dst=False).astimezone(pytz.UTC)
    for img in images:
        try:
            dt = datetime.strptime(img.split(os.sep)[-1][0:15], dtfmt)
            dt = pytz.timezone(ini.tz).localize(dt,is_dst=False).astimezone(pytz.UTC)
            if dt >= sdate and dt <= edate: new_list.append(img)
        except ValueError:
            continue

    return new_list
