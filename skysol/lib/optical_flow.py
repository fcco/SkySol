from __future__ import print_function
import time
import cv2
import numpy as np

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)


class optflow:

    def __init__(self, mask):
        """
        The initialisation of the optical flow algorithm.
        Here settings are made.
        """

        self.p0 = 0
        self.p1 = 0
        self.st = 0
        self.minVectors = 5
        self.maxCorners = 750

        # Mask
        self.mask_no_sun = self.__getMask(mask)

        # Create some random colors for points
        self.color = np.random.randint(0,255,(self.maxCorners,3))

        # params for ShiTomasi corner detection
        self.feature_params = dict( minDistance = 20,
                    blockSize = 30)
        self.qualityLevel = 0.07

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (50,50), maxLevel = 4,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def __getMask(self,mask):
        """
        Use image mask as basis for optical flow masking
        """
        import scipy.ndimage.filters as filt

        # increase the image mask to avoid vectors at the image's edge
        fmask = np.zeros((mask.shape),dtype=np.uint8)
        fmask[mask==False] = 255
        t = filt.minimum_filter(fmask,50)
        fmask[t==0] = 0

        return fmask



    def getCorners(self,img,fmask=""):
        """
        Applies shi-tomasi algorithm from OpenCV to find good features to track

        Input:
        ------
        img: the input image
        fmask: an optional boolean mask (0/1) to mask out e.g. a region in the image

        Returns:
        --------
        p0: array-like
            image coordinates of found features
        """

        # do median blurring before Feature detection
        # in order to remove noise and small artefacts with steep gradients
        # which sometimes are missclassified as features
        #th = cv2.medianBlur(img,9)
        th = cv2.bilateralFilter(img,9,75,75)
        p0 = cv2.goodFeaturesToTrack(th, mask = fmask,
                                         maxCorners = self.maxCorners,
                                         qualityLevel = self.qualityLevel,
                                         **self.feature_params)
        return p0



    def getVectors(self,img1,img2,features, quiet=False):
        """
        Applies Lukas-Kanade method to a set of features from initial image.

        Input:
        ------
        img1: first image
        img2: second image
        features: good points to track from img1

        Returns:
        --------
        new_point: the image coordinates of the corresponding feature findings
            in img2
        old_point: the image coordinates of the raw features from img1
        flow_flag: quality flag (0/1) of each vector
        npoints: number of found vectors
        """

        stime = time.time()
        # input features must be CV_32F
        features = np.float32(features)
        try:
            self.p1, self.st, err = cv2.calcOpticalFlowPyrLK(img1, img2, features, None, **self.lk_params)
            # output must be integer
            new_point = np.int32(np.round(np.array(self.p1[[self.st>=0]]),0))
            old_point = np.int32(np.round(np.array(features[[self.st>=0]]),0))
            flow_flag = self.st.flatten()
            npoints = self.st.shape[0]
        except (cv2.error, TypeError) as e:
            print("Optical Flow vector Lucas-Kanade failed, ", e)
            self.st = 0
            new_point = 0
            old_point = 0
            flow_flag = 0
            npoints = 0
        if not quiet: print('Calculate Optical Flow... finished in %.1f seconds' % (time.time() - stime) )
        if np.sum(self.st) < self.minVectors:
            if type(flow_flag) == int:
                flow_flag = 0
            else:
                flow_flag[:] = 0
        else:
            flow_flag[np.any(new_point>img1.shape[0], axis=1)] = 0
            flow_flag[np.any(old_point>img1.shape[0], axis=1)] = 0

        return new_point, old_point, flow_flag, npoints




    def maskSun(self, spa, theta, horizon, maxdist=np.radians(20)):

        """
        Function to mask the region around sun

        Input:
        ------
        mask - image mask
        spa - sun pixel angle array (radians, angular distance to sun for each pixel)
        theta - the image incidence angle array
        horizon - the image mask horizon
        maxdist - angular distance in pixel (radians) from sun for region to mask

        Output:
        fmask - mask image
        """
        fmask = self.mask_no_sun.copy()

        # mask region around sun
        fmask[spa < maxdist] = 0

        # mask horizon
        fmask[theta >= np.radians(horizon-2.0)] = 0

        return fmask



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


def predict(img,u,v,x):

    #import matplotlib.pyplot as plt

    print(img.shape)
    print(u.shape)

    new_img = np.zeros_like(img)

    map_x = np.mgrid[0:img.shape[0],0:img.shape[1]]
    map_y = np.mgrid[0:img.shape[0],0:img.shape[1]]
    map_x = np.array(map_x - x,dtype=np.float32)
    map_y = np.array(map_y - u,dtype=np.float32)

    new_img = cv2.remap(img,map_x,map_y,cv2.INTER_NEAREST)

    #new_img = img[]

    return new_img



def drawVectors(flags,spoints,epoints,mask, vcolor=[0,0,255], vsize=6, skip=4, mask_flag=True):

    st = 1
    arrow_magnitude = vsize * 2
    if np.sum(st) != 0 and type(spoints) != np.int:
        try:
            # draw the tracks and arrows
            for i,(new,old) in enumerate(zip(spoints,epoints)):
                if flags[i] == True:

                    if i % skip != 0:
                        continue

                    a,b = new.ravel()
                    c,d = old.ravel()

                    a = int(a); b = int(b); c = int(c); d = int(d)
                    cv2.line(mask,(a,b),(c,d) , vcolor, vsize)
                    cv2.circle(mask,(a,b),int(vsize)-1,vcolor,-1)
                    # calc angle of the arrow
                    angle = np.arctan2(a-c, b-d)
                    # starting point of first line of arrow head
                    p = (int(c - arrow_magnitude * np.cos(angle + np.pi/4)),
                         int(d + arrow_magnitude * np.sin(angle + np.pi/4)))

                    # draw first half of arrow head
                    cv2.line(mask, p, (c,d),  vcolor, vsize)
                    # starting point of second line of arrow head
                    p = (int(c + arrow_magnitude * np.cos(angle - np.pi/4)),
                         int(d + arrow_magnitude * -np.sin(angle - np.pi/4)))
                    # draw second half of arrow head
                    cv2.line(mask, p, (c,d), vcolor, vsize)
        except (TypeError) as e:
            print(e)
            pass

    return mask
