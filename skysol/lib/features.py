import numpy as np
import math
import csv
import scipy.stats
import cv2
import skimage.feature
import time
import os
import gc

class features:

    def __init__(self, mask, theta):

        self.vec = np.empty(27,dtype=np.float64)
        self.vec[:] = np.nan
        self.contour_img = None
        self.mask = self.__getMask(mask, theta)
        self.names = ["Contrast",
        "Dissimilarity", "Homogenity", "Energy", "Correlation", "ASM", "Mean_Red" , \
        "Mean_Blue", "Difference_Red-Green", "Difference_Red-Blue", \
        "Difference_Green-Blue", "Standard_Deviation_Blue", "Skewness_Blue", \
        "Cloud_Coverage", "Mean_gray_value", \
        "Ratio_of_saturated_pixels_to_non-masked_pixels(RGB)", \
        "Ratio_of_saturated_pixels_to_non-masked_pixels(HSV)", \
        "Overall_RB_ratio", \
        "Cloud_coefficient", "Gray_coefficient", "Mean_object_area", "Mean_object_extent",  "Mean_object_perimeter", \
        "Mean_object_solidity", "Mean_object_intensity", \
        "Mean_object_equivalent_diameter", "Number_of_objects"]


        # Feature Order:
        # Contrast
        # Dissimilarity
        # Homogenity
        # Energy
        # Correlation
        # ASM
        # Mean Red
        # Mean Blue
        # Difference Red-Green
        # Difference Red-Blue
        # Difference Green-Blue
        # Standard Deviation Blue
        # Skewness Blue
        # Cloud Coverage ( Ratio of cloudy to non-cloudy pixels )
        # Mean gray value (intensity)
        # Ratio of saturated pixels to all non-masked pixels (RGB)
        # Ratio of saturated pixels to all non-masked pixels (Hue)
        # Overall R/B ratio ( average over all non-masked pixels )
        # Cloud coefficient
        # Gray coefficient
        # Mean object area
        # Mean object extent
        # Mean object perimeter
        # Mean object solidity
        # Mean object intensity
        # Mean object equivalent diameter
        # Number of objects



    def __getMask(self,mask, theta):
        """
        Masking image region for which features should be computed
        """
        fmask = np.zeros((mask.shape),dtype=np.uint8)
        # objects
        fmask[mask==False] = 255
        # horizon
        fmask[theta > np.radians(80) ] = 0
        return fmask


    def writeVector(self,ini,filename,actdate,cnt):

        # date and time information
        dt = list(actdate.utctimetuple())[0:6]
        # features
        fmt = '%.7f'
        outlist = dt + [ fmt % h for h in self.vec ]
        if (ini.append or cnt > 1) and (os.path.exists(filename)):
            with open(filename, "a") as f:
                print(*outlist,sep=" ",file=f)
        else:
            headerline = ["Year", "Month", "Day", "Hour", "Minute", "Second"] + self.names
            with open(filename, "w") as f:
                print(*headerline, sep=" ", file=f)
                print(*outlist, sep=" ", file=f)




    def read_vector(vector):

        reader = csv.reader(open(filename, "r"),delimiter=" ")
        for row in reader:
            vector.append(row)




    def get_features(self,ini,image,cloud_cover, cloud_map, cloud_image, cam, contour_flag=False,draw_contours=False):


        imgsize = image.shape

        img = image.orig_gray.copy()
        img[self.mask==0] == 0
        # Scale gray color from 256 to 64
        new_gray = 64. * img  / 255.

        # Convert to HSV channels
        hsv = cv2.cvtColor(image.orig_color,cv2.COLOR_BGR2HSV)

        # Co-occurence-Matrix
        Ma=np.empty([65,65])

        # Calculate co-occurence mattrix with distance 1 and angle 0 degrees
        g = skimage.feature.greycomatrix(img, [1], [0],levels=256,normed=True,symmetric=True)


        self.vec[0] = skimage.feature.greycoprops(g, 'contrast')
        self.vec[1] = skimage.feature.greycoprops(g, 'dissimilarity')
        self.vec[2] = skimage.feature.greycoprops(g, 'homogeneity')
        self.vec[3] = skimage.feature.greycoprops(g, 'energy')
        self.vec[4] = skimage.feature.greycoprops(g, 'correlation')
        self.vec[5] = skimage.feature.greycoprops(g, 'ASM')

        R = np.empty([imgsize[0],imgsize[1]])
        G = np.empty([imgsize[0],imgsize[1]])
        B = np.empty([imgsize[0],imgsize[1]])
        R = image.orig_color[:,:,2]
        G = image.orig_color[:,:,1]
        B = image.orig_color[:,:,0]

        # Mean Red
        a = cv2.mean(R,mask=self.mask)
        self.vec[6] = a[0]


        # Mean Blue
        a = cv2.mean(B,mask=self.mask)
        self.vec[7]= a[0]
        a = cv2.mean(G,mask=self.mask)
        Gm = a[0]


        # Difference ( Red - Green )
        self.vec[8]=(self.vec[6]-Gm)


        # DIfference ( Red - Blue )
        self.vec[9]=(self.vec[6]-self.vec[7])


        # Difference ( Green - Blue )
        self.vec[10]=(Gm-self.vec[7])


        # Standard Deviation ( Blue )
        sdB=np.std(B[self.mask==255])
        self.vec[11]=sdB


        # Skewness ( Blue )
        bb = scipy.stats.skew(B[self.mask==255])
        self.vec[12]= bb


        # Cloud Coverage
        self.vec[13] = cloud_cover


        # Mean gray value
        self.vec[14] = np.mean(img[self.mask==255])


        # Ratio of saturated pixels to all non-masked pixels
        nsatpix = np.sum(img >= 254.)
        nnonmaskpix = np.sum(self.mask==255)
        self.vec[15] = float(nsatpix) / float(nnonmaskpix)


        # Ratio of saturated pixels in hue channel to all non-masked pixels
        nsatpix = np.sum(hsv[:,:,0] == 0)
        self.vec[16] = float(nsatpix) / float(nnonmaskpix)


        # Overall R/B ratio
        self.vec[17] = np.nanmean(image.rbr[self.mask==255])

        # Cloud coefficient
        imggrid = np.mgrid[0:cloud_map.shape[0],0:cloud_map.shape[1]]
        imggrid = ( imggrid - imggrid.shape[1]/2 ) / float(imggrid.shape[1]/2)
        weight = 1 - np.sqrt(imggrid[0]**2 + imggrid[1]**2) # pixels are weighted with distance to center
        cmap = np.float16(cloud_map)
        # define clouds ( mask = NAN, sky = 0, cloud = 1 )
        cmap[cloud_map==0] = np.nan
        cmap[cloud_map==1] = 0
        cmap[cloud_map>1] = 1
        # get sum
        ccm = np.multiply(cmap,weight)
        cc = np.nansum(ccm)
        # all not-NAN pixels
        allp = np.sum(np.isfinite(ccm))
        # coefficient
        self.vec[18] = (cc / float(allp))


        # Gray coefficient
        weight = np.cos(cam.theta) # pixels are weighted with distance to center
        gmap = cloud_image.copy()
        # define gray values ( mask = NAN, sky = 0, cloud = 1 )
        gmap[cloud_image==0] = np.nan
        gmap[cloud_image==1] = 0
        gmap[cloud_image>1] = gmap[cloud_image>1] / 255.
        # get sum
        gcm = np.multiply(gmap,weight)
        gcc = np.nansum(gcm)
        # all not-NAN pixels
        allp = np.sum(np.isfinite(gcm))
        # coefficient
        self.vec[19] = (gcc / float(allp))

        del img, imggrid, cmap

        if contour_flag:

            if draw_contours:
                self.contour_img = image.orig_color_draw

            # Find contours on segmented image
            th = image.binary_gray.astype(np.uint8).copy()
            th[th>1] = 255
            th[th<255] = 0
            _, contours, hierarchy = cv2.findContours(th.copy(),\
                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


            areal = []; perimeter = []; extent = []; solidity = []
            equi_diameter = []; mean_val = []

            objcnt = 0

            for i in range(1,len(contours)):

                cnt = contours[i]

                # sort small objects out
                if len(contours[i]) <= 50:
                    continue
                objcnt += 1

                mask = np.zeros(image.shape,np.uint8)
                cv2.drawContours(mask,[cnt],0,255,-1)
                pixelpoints = np.transpose(np.nonzero(mask))

                # Object Area
                area = (cv2.contourArea(cnt)) / 1000.
                areal.append(area)

                # Object Perimeter
                perimeter.append(cv2.arcLength(cnt,True))

                # Extent: Ratio of Contour Area to Bounding Rect
                area = cv2.contourArea(cnt)
                x,y,w,h = cv2.boundingRect(cnt)
                rect_area = w*h
                extent.append(float(area)/rect_area)

                # Solidity: Ratio of Contour Area to Convex Hull
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity.append(float(area*1000.)/hull_area)
                else:
                    solidity.append(np.nan)

                # Equivalent Diameter: Diameter of the circle whose area is the same as the contour area
                equi_diameter.append(np.sqrt(4*area/np.pi))

                # Mean Color of the object
                mean_val.append(cv2.mean(image.orig_color,mask = mask))

                # draw contours
                if draw_contours:
                    cv2.drawContours(self.contour_img,cnt,-1,(100,100,0),3)

            self.vec[20] = np.mean(areal)
            self.vec[21] = np.mean(extent)
            self.vec[22] = np.mean(perimeter)
            self.vec[23] = np.mean(solidity)
            self.vec[24] = np.mean(mean_val)
            self.vec[25] = np.mean(equi_diameter)
            self.vec[26] = objcnt
            del th

        gc.collect()