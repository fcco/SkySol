#!/usr/bin/env python
"""
title	       : SkySol
description	   : Sky Imager based Solar Irradiance Analyses and Forecast Tool
author		   : Thomas Schmidt
date	       : 20141218
last change    : 20181228
version		   : 0.5
usage		   : python3 main.py
notes		   :
python_version : 3.5
==============================================================================
"""
__author__ = "Thomas Schmidt"
__copyright__ = "Copyright 2014, University of Oldenburg"
__version__ = "0.5"
__maintainer__ = "Thomas Schmidt"
__email__ = "t.schmidt@uni-oldenburg.de"
__status__ = "development"


if __name__ == "__main__":


    print("##########################################")
    print("#             SkySol                     #")
    print("#                                        #")
    print("# Sky Imager Analysis and Forecast Tool  #")
    print("#                                        #")
    print("#           Version "+__version__+"                  #")
    print("#                                        #")
    print("##########################################")

    # Import System Tools
    import sys
    import gc
    import glob
    import shutil
    import os

    # import date and time tools
    from datetime import datetime, timedelta
    import time
    import calendar
    import pytz

    # Handle Input arguments
    from skysol.lib.misc import args
    inargs = args(sys.argv[1:])

    # Import user defined settings
    from skysol.settings import settings
    ini = settings.ini(indate=inargs.indate)

    # Handle different modes ("Live", "Archive", "Single")

    if inargs.imgpath != "":
        ini.imgpath = inargs.imgpath
        ini.mode = 1
        ini.live = False
    if inargs.mode == 'live':
        ini.actdate = datetime.utcnow()
        inargs.sdate = ini.actdate.replace(hour=0, minute=0)
        ini.live = True
        ini.mode = 0
        print('Running SkySol in Live mode')
    elif inargs.mode == 'single':
        ini.mode = 1
        ini.live = False
        print('Running SkySol in single picture mode')
    elif inargs.mode == 'archive':
        ini.mode = 0
        ini.live = False
        print('Running SkySol in archive mode')
    else:
        sys.exit("Specify mode in first argument: 'archive','live', 'single'")


    if ini.log_flag:
        # Redirect all prints to this log file
        if not os.path.exists(ini.outdir + os.sep + 'log'):
            os.makedirs(ini.outdir + os.sep + 'log')
        print("Output will be written to log file %s" % ini.logfile)
        sys.stdout = open(ini.logfile,'a')
        sys.stderr = open(ini.logfile,'a')

    # import opencv
    import cv2

    ### TODO: live image processing not yet tested (use images from local directories instead)
    if ini.live and ini.stream_flag:
        print("Get Camera Stream...")
        http_proxy = ""
        cap = cv2.VideoCapture(ini.http)
    else:
        cap = ""

    # import NumPy
    import numpy as np
    from numpy import radians

    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')

    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", FutureWarning)

    # Import Matplotlib for visualization
    #import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    # load local modules
    from skysol.lib import misc # miscellanoues methods
    from skysol.lib import radiation # radiation measurement handling
    from skysol.lib import camera_cmv# cloud motion vectors
    from skysol.lib import cloudbase # cloudbase handling
    from skysol.lib import image # image handling
    from skysol.lib import camera_matrix # camera characteristics
    from skysol.lib import cloud_decision # cloud decision algorithms
    from skysol.lib import optical_flow # optical flow
    from skysol.lib import visualization # result plots
    from skysol.lib import forecast # forecasting methods
    from skysol.lib import clearlib # clear sky image library
    from skysol.lib import solar # solar geometry library
    from skysol.lib import clearsky # clear sky radiation library
    from skysol.lib import features # image feature extraction
    from skysol.lib import classification # cloud type classification
    from skysol.lib import output # handle written output
    from skysol.lib import drawings # graphical toolbox
    from skysol.lib import pvsystem # PV system library


    # Measure timing
    startstart_time = time.time()

    # Check settings
    misc.check_settings(ini)
    if ini.log_flag: misc.dump(ini)

    counter = 0
    print("Initialise software...")

    # Initialise Camera Matrix class
    st = time.time()
    cam = camera_matrix.camera(ini)
    print("Camera model...\t\tfinished in %.1f seconds" % round(time.time() - st, 1))


    # Initialise Clear Sky Library
    st = time.time()
    csl = clearlib.csl(ini.cslfile, read=ini.csl_flag)
    if ini.csl_flag:
        print("Clear sky library...\tfinished in %.1f seconds\t" % round(time.time() - st, 1))

    # Initialise first image
    st = time.time()
    if ini.mode == 0:
        sdir = ini.picpath + os.sep + ini.locdate_str + os.sep + '*' + ini.imgformat
        images = sorted(glob.glob(sdir))
        images = image.filter(ini, inargs.sdate, images, edate=inargs.edate)
    elif ini.mode == 1:
        sdir = inargs.imgpath + os.sep + '*.' + ini.imgformat
        images = sorted(glob.glob(sdir))
    if len(images) == 0: sys.exit('no images found, check the search directory ' + sdir + ' for files. -> Exit!')

    if ini.live:
        img0 = image.image(ini, images, -999, cap=cap)
    else:
        img0 = image.image(ini, images, 0, cap=cap)
    filename = img0.filename

    # get datetime information for current image
    actdate = img0.actdate
    print("Read image...\t\tfinished in %.1f seconds\t%d images found" % (round(time.time() - st, 1),len(images)))

    # Initialise radiation data
    ini.data_file = datetime.strftime(img0.locdate,ini.data_fmt)
    nstations, pyr = radiation.iniRadData(ini,locations=ini.flocs)

    # Image background mask
    mask_bool = image.get_mask(ini,img0.orig_gray,cam.dtc_norm)
    mask_horizon = cam.theta > np.radians(ini.horizon)
    cam.dtc_norm = 0; del cam.dtc_norm
    img0.mask_bool = mask_bool
    img0.orig_gray[mask_bool] = 0

    # Initialise cloud base data
    st = time.time()
    cld = cloudbase.cloudbase(ini)
    print("Read cloud heights...\tfinished in %.1f seconds" % round(time.time() - st, 1))
    # Current sun position
    sun = solar.compute([actdate], ini.lat0, ini.lon0, method="spencer")
    sza = sun['zenith'][0]; saz = sun['azimuth'][0]

    # Compute sun position in image coordinates
    xsun, ysun = cam.sphere2img(sza, saz, rotate=True)

    # Compute sun pixel angle matrix ( angular distance of pixel to sun )
    spa = misc.sunpixelangle( cam.theta, cam.phi2, sza, saz )

    # Optical Flow: Initialize settings for Optical flow
    flow = optical_flow.optflow(mask_bool)

    # Optical Flow: Mask for feature detection (exclude horizon region + sun region)
    fmask = flow.maskSun(spa, cam.theta, ini.horizon, maxdist=np.radians(20))

    # Optical Flow: Find features to track
    flow.p0 = flow.getCorners(img0.orig_gray, fmask) # Get corners

    # Initialise Cloud Motion Vectors
    cmv = camera_cmv.cmv(ini)

    # Create a mask image for drawing purposes
    cmv_mask = np.zeros_like( img0.orig_color )

    # Initialise feature class
    features = features.features(mask_bool, cam.theta)

    # Datetime instances for current day
    if ini.live or (len(pyr[0].dates) == 0):
        dates = [ pytz.timezone(ini.tz).localize(datetime(actdate.year, \
            actdate.month, actdate.day)).astimezone(pytz.utc) + \
            timedelta(seconds=i) for i in range(0,86400) ]
    else:
        dates = pyr[0].dates

    # Compute and get clear sky reference for current day
    csk = clearsky.clearsky(dates, ini.lat0, ini.lon0, source="pvlib", model="simplified_solis")

    # Compute solar position for whole day
    sun = solar.compute(dates, ini.lat0, ini.lon0, method="spencer")
    csk.cossza = np.cos(sun['zenith'][:])

    # Initialise dictionary for meta data
    meta = {}

    # Learn Cloud Classification Model
    cc = classification.classificator(ini)
    if ini.cloud_class_learn:
        cc.get_training(ini)
        cc.fit(ini.cloud_class_model, model=ini.class_model, \
            rank_feat=ini.rank_features, print_feat=ini.print_features)

    # image counter
    cnt = 1

    hdfinit = True
    ana_mbe = 0.0


    # Initialise mapping matrices for undistortion
    map_x, map_y = cam.backward(ini, cld.cbh, saz, sza, \
            cbh_flag=ini.cbh_flag, limit=ini.horizon, \
            shadow_flag=ini.shadow_flag)


    # and from perspective to fish eye
    map_inv_x, map_inv_y = cam.forward(ini.horizon)

    qflag = True
    clr_ref = img0.orig_color
    orig_gray_old = img0.orig_gray
    orig_color_old = img0.orig_color
    sza_old = np.radians(180)
    del img0
    gc.collect()

    # loop over all frames
    print("")
    print("Start loop...")
    while(1):

        start_time = time.time()

        if ini.log_flag:
            sys.stdout.close(); sys.stderr.close()
            sys.stdout = open(ini.logfile,'a'); sys.stderr = open(ini.logfile,'a')
        gc.collect()

        #==================
        #
        # Read next image
        #
        #==================
        if ini.live:
            img = image.image(ini, images, -1, cap=cap, filename = filename)
        else:
            img = image.image(ini, images, cnt, cap=cap)
        print("Next image")

        img.mask_bool = mask_bool
        img.mask_horizon = mask_horizon

        cnt += 1

        # process next image
        if cnt % ini.camera_res == 0 and img.exit == 1:

            # convert some date and time variables
            lastdate = actdate
            locdate = img.locdate
            actdate = img.actdate
            actnumdate = np.int64(calendar.timegm(img.actdate.utctimetuple()))
            newdate_str = datetime.strftime(img.locdate, "%Y%m%d")
            plotstring = datetime.strftime(img.actdate, "%Y-%m-%d %H:%M:%S UTC")
            filename = img.filename
            cmv.lastimg = (actdate - lastdate).total_seconds()

            # copy of original image for drawing purposes
            img.orig_color_draw = img.orig_color.copy()

            # Stop after day completion
            if (newdate_str != str(ini.locdate_str) and ini.mode != 1):
                print('Day completed! -> exit')
                break

            csl = clearlib.csl(ini.cslfile, read=ini.csl_flag)

            #===================================================================
            # Sun Position
            #===================================================================
            sun = solar.compute([actdate], ini.lat0, ini.lon0, method="spencer")
            sza = sun['zenith'][0]
            saz = sun['azimuth'][0]

            xsun, ysun = cam.sphere2img(sza, saz, rotate=True)

            # Global image-size array with angular distance to sun in raw image
            spa = misc.sunpixelangle( cam.theta, cam.phi2, sza, saz )

            print("Date and Time:\t\t%s Sun Elevation: %.1f deg Azimuth %.1f deg" % \
                (img.locdate, 90-np.degrees(sza), np.degrees(saz)) )

            # Skip image if sun is below user defined limit
            sun_elev = 90-np.degrees(sza)
            sun_grad = sza - sza_old
            sza_old = sza
            # Morning
            if (sun_elev < ini.min_sun_elev) and sun_grad > 0 and ini.mode == 0:
                print("Sun elevation (%.1f) below limit (%.1f). -> GoodBye!\n" \
                      % (sun_elev, ini.min_sun_elev) )
                sys.exit(2)
            # Evening
            if (sun_elev < ini.min_sun_elev) and sun_grad < 0:
                print("Sun elevation (%.1f) still below limit (%.1f). -> Skip image!\n" \
                      % (sun_elev, ini.min_sun_elev) )
                continue
            if ini.fcst_flag and (np.degrees(sza) > ini.horizon-0.1):
                print("Sun elevation (%.1f) below image horizon limit (%.1f). -> Skip image!\n" \
                    % (sun_elev, 90-ini.horizon) )
                continue
            #-------------------------------------------------------------------



            #===================================================================
            # NRT data update for current image ( in live mode / new day )
            #===================================================================
            st = time.time()
            if ini.live:
                pyr[0].read_live(ini.nrt_rad)
                cld = cloudbase.cloudbase(ini)

            if ini.mode == 1 and ini.radiation:
                ini.data_file = datetime.strftime(img.locdate,ini.data_fmt)
                nstations, pyr = radiation.iniRadData(ini)
                dates = np.array([ datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
                    for ts in pyr[0].time ])
                csk = clearsky.clearsky(dates, ini.lat0, ini.lon0, source="pvlib")
            if ini.live or ini.mode == 1:
                print("NRT data update...\tfinished in %.1f seconds" % (time.time() - st))
            #-------------------------------------------------------------------



            #===================================================================
            # Cloud base height
            #===================================================================
            if cld.file_flag:
                cld.cbh = cld.current(actnumdate, cld.cbh, ini.def_cbh)
                print('Current cloud height reading: %d' % cld.cbh)
            else:
                cld.cbh = ini.def_cbh  # Default height
            meta['cbh'] = cld.cbh
            #-------------------------------------------------------------------


            #===================================================================
            # Image masking
            #
            # Boolean arrays
            # mask_horizon: masking a circular horizon
            # mask_bool:    user generated static mask
            # img_mask:     dynamic artifical object mask (birds, humans, ...)
            # mask:         all masks
            #===================================================================


            # apply mask on frame
            img.orig_gray[mask_bool] = 0
            img.mask_horizon = mask_horizon

            st = time.time()

            img_flag = img.check_quality(mask_bool)
            if ini.quality_check:
                img_mask, qflag, clr_ref = img.artifical_object_detection(orig_color_old, qflag, clr_ref, resize=(600,600))
                print("Quality check...\tfinished in %.1f seconds\tBirds %d, Droplets %d" % \
                ( round(time.time() - st,1),img.area_bird, img.area_drop))
            else:
                img_mask = mask_bool
                new_mask = mask_bool
                img.new_mask = new_mask
                qflag = -1
                img.useful_image = -1
            meta['lens'] = qflag
            meta['img_qc'] = img.useful_image

            mask = mask_bool | new_mask | mask_horizon
            if not img_flag:
                print("Image too bright ( overexposed? ) -> Skip image"); continue


            #===================================================================
            # Cloud (shadow) mapping (for current sun position and cloud height)
            #===================================================================

            if ini.cbh_flag or ini.shadow_flag:
                st = time.time()
                map_x, map_y = cam.backward(ini, cld.cbh, saz, sza, \
                    cbh_flag=ini.cbh_flag, limit=ini.horizon, \
                    shadow_flag=ini.shadow_flag)
                print("Image projection...\tfinished in %.1f seconds" % \
                    (time.time() - st))


            #===================================================================
            # Clear sky data ( get current index )
            #===================================================================
            st = time.time()
            csk.tind = misc.get_index(csk.time, actdate) # get index
            if csk.tind == -1: csk.actval = 0.0
            csk.actval = csk.ghi[csk.tind]


            #===================================================================
            # Radiation measurements
            #===================================================================
            if ini.radiation:
                for i in range(0, nstations):
                    # current index
                    if not pyr[i].mflag: continue
                    pyr[i].tind = misc.get_index(pyr[i].time,actnumdate)
                pi = pyr[0].tind


                # statistics for camera position (pyranometer 0)
                if ini.live:
                    radtxt = "last 1min average"
                    ghi, dhi, dni = radiation.sub_series(pyr[0],60)
                    cghi, cdhi, cdni = radiation.sub_series(csk,60)
                else:
                    radtxt = "last "+str(ini.rate)+"s average"
                    ghi, dhi, dni = radiation.sub_series(pyr[0],ini.rate)
                    cghi, cdhi, cdni = radiation.sub_series(csk,ini.rate)
                csi_ghi = np.nanmean(ghi / cghi)
                if len(dhi) > 0:
                        csi_dhi = np.nanmean(dhi / cdhi)
                        csi_dni = np.nanmean(dni / cdni)
                else:
                        csi_dhi = np.nan
                        csi_dni = np.nan
            else:
                # set radiation variables to NaN
                ghi = np.nan; dhi = np.nan; dni = np.nan; radtxt = "not available"
                csi_ghi = np.nan; csi_dhi = np.nan; csi_dni = np.nan
            #-------------------------------------------------------------------



            #===================================================================
            # RBR calculations
            #===================================================================
            st = time.time()
            img.rbr = img.compute_rbr(img.orig_color, mask_bool)
            # img.rbr_orig will be the raw RBR array
            img.rbr_orig = img.rbr.copy()

            # RBR corrections
            if ini.csl_flag and csl.csl_flag:
                csltype = img.rbr_csl_correction(ini, cam, csl, actnumdate, saz, sza, mask_bool)
            else:
                csltype = 0
            print("RBR calculation...\tfinished in %.1f seconds" % (time.time() - st) )
            #-------------------------------------------------------------------

            #===================================================================
            # Cloud detection and Cloud cover
            #===================================================================
            st = time.time()

            # Segment clouds based on RBR
            img.segment_clouds(ini.rbr_thres)

            # Compute binary images based on segmentation
            img.make_binary(mask_bool)

            # Apply a median Filter on binary image to remove noise
            img.median_filter(ini.median_filter, mask_bool)

            # Cloud cover
            tmp = img.binary_gray.copy()
            # Cloud Cover of fisheye image (without mask)
            cc_img = cloud_decision.cloud_cover(tmp)
            cc_img_un = cloud_decision.cloud_cover(cam.grid(tmp,map_x,map_y))
            # Cloud cover with sun masked out
            tmp[spa < np.radians(ini.sunspot) ] = 0
            cc_img_wo_sun = cloud_decision.cloud_cover(tmp)

            print("Cloud detection...\tfinished in %.1f seconds\tCC = %d%% (%d%%)" % (time.time() - st, cc_img, cc_img_wo_sun) )
            #-------------------------------------------------------------------

            #===================================================================
            # Feed clearsky library (based on cloud cover and/or DNI measurements)
            #==================================================================
            if not ini.live and ini.write_csl:
                st = time.time()
                wrt_flag = False

                # Clear sky detection based on user input (user specifies images to be clear)
                if inargs.isclear:
                    wrt_flag = csl.write_db(actnumdate, sza, saz)
                else:
                    # If DNI data available, use a DNI threshold together with CC for clear sky detection
                    if ini.radiation and len(pyr[0].dni) > 0:
                        ghi, dhi, dni = radiation.sub_series(pyr[0],15)
                        cghi, cdhi, cdni = radiation.sub_series(csk,15)
                        csi_ghi = np.nanmean( ghi / cghi )
                        csi_dni = np.nanmean( dni / cdni )
                        csi_dhi = np.nanmean( dhi / cdhi )
                        if cc_img_wo_sun < 10 and abs(csi_dni) > 0.8:
                            # store clear sky day to csl
                            wrt_flag = csl.write_db(actnumdate, sza, saz, cc)
                    else:
                        # Clear sky detection based on CC threshold
                        if (cc_img_wo_sun < 10):
                            wrt_flag = csl.write_db(actnumdate, sza, saz)

                if wrt_flag:
                    print("Feeding CSL...\t\tfinished in %.1f seconds\t image added" % (time.time()-st))
            #-------------------------------------------------------------------



            #===================================================================
            # Cloud Map
            #===================================================================
            binary_image = img.binary_gray.copy()
            binary_image[mask] = 0 # mask

            #===================================================================
            # Image features
            #===================================================================
            circ = -1; gray = -1
            meta['gray'] = gray
            meta['circ'] = circ
            featvec = []
            vimgfeats = {}

            if ini.features:

                st = time.time()

                # Define final undistorted cloud map ( mask = 0, sky = 1, cloud > 1 )
                # for cloud coefficients
                cimg = binary_image.copy().astype(np.float32)
                cimg[spa<np.radians(3)] = np.nan
                cimg[mask] = np.nan
                cmap = cam.grid(cimg,map_x,map_y)

                # Compute image features
                features.get_features(ini, img, cc_img, cmap, cimg, cam, \
                    contour_flag = ini.contour_flag, draw_contours = ini.draw_contours)

                if ini.contour_flag and ini.draw_contours:
                    img.orig_color_draw = features.contour_img

                # Gray coefficients
                gray = cloud_decision.gray_coeff(cimg,cam, spa)

                # Circumsolar area
                circ = {}
                for i in [ 0.5, 1.5, 3, 5, 7, 10, 15, 20 ]:
                    circ[str(i)] = cloud_decision.sun_area(img,cam, sza, saz, sunarea=i)

                meta = dict( time = actnumdate, cloud_cover = cc_img,  \
                    cloud_cover_stretched = cc_img_un,
                    cloud_cover_wo_sun = cc_img_wo_sun,
                    ghi_avg = np.nanmean(ghi),
                    ncmv = cmv.npoints,
                    sun_azimuth = np.degrees(saz),
                    sun_zenith = np.degrees(sza),
                    gray = gray,
                    sunspot = circ['3'],
                    cbh = cld.cbh,
                    lens = qflag,
                    img_qc = img.useful_image
                    )
                meta_rad = dict ( circ = circ, gray = gray, cossza = np.cos(sza))

                # Merge all features for irradiance estimation
                imgfeats = dict(zip(features.names, features.vec))

                if ini.write_meta and ini.features:
                    if ini.mode == 0:
                        outfile = ini.outdir + '/meta/' + ini.locdate_str + '.features'
                    elif ini.mode == 1:
                        outfile = ini.outdir + '/meta/features.dat'
                    features.writeVector(ini, outfile, actdate, cnt-2)

                print("Feature extraction...\tfinished in %.1f seconds" % (time.time() - st))




            #===================================================================
            # Radiation modeling
            #===================================================================
            dni_level = 1.0

            # compute the percentage of saturated pixels in the sunspot region
            # defined by radius "sunarea" (degrees)
            sun_vis = cloud_decision.sun_area(img, cam, sza, saz, sunarea=3)


            if ini.radiation:
                st = time.time()

                # defaults
                dni_level = 1.0; pyr[0].csi_max = 1.0; pyr[0].csi_min = 0.4

                # current GHI
                ghi = pyr[0].ghi[pi]
                # current Clear Sky GHI
                cghi = csk.ghi[csk.tind]

                # Archive/Forecasting mode
                if ini.mode == 0 :

                    # Irradiance modelling (computed current GHI CSI and lower
                    # and upper range for shadow to irradiance mapping
                    csi = radiation.getCSI(ini, nstations, pyr, csk, sza, featvec, \
                                           mode=ini.csi_mode)

                    # Compute DNI from measured GHI and estimated DHI
                    tdni  = ( pyr[0].ghi[pi] - pyr[0].csi_min * csk.ghi[csk.tind] ) / np.cos(sza)
                    # Clear sky index of DNI
                    pyr[0].dni_csi_now = tdni / csk.dni[csk.tind]

                    #================
                    # Bias correction
                    #================
                    if ini.bias_correction and (sun_vis == 0):

                        # If sun blocked, no DNI is assumed -> compute DHI bias
                        dhi_bias = pyr[0].csi_min - (ghi / cghi)

                        # correct lower irradiance level for bias
                        pyr[0].csi_min = pyr[0].csi_min - dhi_bias
                        pyr[0].dni_csi_now = 0.0

                    #===========
                    # DNI clear
                    #===========

                    # Estimate the clear sky direct irradiance level from previous
                    # measurements in clear sky conditions
                    dni_est = radiation.dni_clear( pyr[0], csk, nimg=90, min_data=6)
                    # get maximum (average of values > x percentile) DNI level from last 30 minutes
                    dni_level = radiation.dni_level(dni_est, percentile=0.9, percentage_valid=0.1, lower_limit=0.4)

                    # adapt CSI upper limit to the DNI level
                    if dni_level > 0:
                        pyr[0].csi_max = pyr[0].csi_min + ( dni_level * csk.dni[csk.tind] * np.cos(sza) / csk.actval )

                    print("Irradiance modeling\tfinished in %.1f seconds\tDNI Level: %.2f, Bias Correction: %.2f, k* range: %.2f/%.2f" % \
                        (time.time() - st, dni_level,ana_mbe,pyr[0].csi_min,pyr[0].csi_max) )

                # In single picture mode
                else:
                    # set defaults
                    pyr[0].csi_max = 1.0; pyr[0].csi_min = 0.4

                    # Bias correction
                    pyr[0].csi_max = pyr[0].csi_max - ana_mbe
                    pyr[0].csi_min = pyr[0].csi_min - ana_mbe
                    pyr[0].dni_csi_now = 1

                # apply to other stations
                for i in range(0,nstations):
                    pyr[i].csi_max = pyr[0].csi_max
                    pyr[i].csi_min = pyr[0].csi_min

            # no irradiance values available
            else:
                # Compute DNI from default CSI low and CSI high
                tdni  = ( csk.ghi[csk.tind] - pyr[0].csi_min * csk.ghi[csk.tind] ) / np.cos(sza)
                # Clear sky index of DNI
                pyr[0].dni_csi_now = tdni / csk.dni[csk.tind]

            #===================================================================
            # Cloud type classification
            #===================================================================
            if ini.cloud_class_apply == 1:
                st = time.time()
                # Apply Cloud Classification Model to unclassified image
                cloud_class, cloud_class_prob = cc.apply_model(imgfeats, ini.cloud_class_model, rank_feat=ini.rank_features)
                if type(cloud_class_prob) != float:
                    print("Cloud Classification...\tfinished in %.1f seconds\tClass: %d, Prob: %.2f" % \
                        (time.time() - st, cloud_class, cloud_class_prob[cloud_class-1]))
            else:
                cloud_class = -1; cloud_class_prob = np.nan

            meta['CloudClass'] = cloud_class
            meta['CloudClassProb'] = cloud_class_prob

            # add cloud type classification results to lists
            cc.cloud_class.append(cloud_class); cc.class_prob.append(cloud_class_prob);
            cc.time.append(actnumdate)







            #===========================================================================
            # Cloud Motion
            #===========================================================================

            # Apply Optical Flow
            if ini.flow_flag:

                st = time.time()

                # Renew corners at given interval
                if counter >= int(ini.renew_cmv / ini.rate) or cmv.lastimg > 60:

                    # Mask sun area
                    fmask = flow.maskSun(spa, cam.theta,
                                         ini.horizon, maxdist=np.radians(20))

                    # Reset variables
                    counter = 0
                    cmv_mask = np.zeros_like( img.orig_color )

                    flow.p0 = flow.getCorners( img.orig_gray,fmask) # Get corners
                    cmv.old_u = []; cmv.old_v = []; cmv.old_flag = []

                # Apply Lucas-Kanade and return vector coordinates
                cmv.new_point, cmv.old_point, cmv.flow_flag, cmv.npoints = \
                    flow.getVectors(orig_gray_old, \
                        img.orig_gray, flow.p0, quiet=True)

                if cmv.npoints > 0:
                    gv = cmv.flow_flag[:] == 1

                    # CMV in spherical coordinates
                    prevzenith = cam.theta[cmv.old_point[gv,0],cmv.old_point[gv,1]]
                    prevazimuth = cam.phi2[cmv.old_point[gv,0],cmv.old_point[gv,1]]
                    currzenith = cam.theta[cmv.new_point[gv,0],cmv.new_point[gv,1]]
                    currazimuth = cam.phi2[cmv.new_point[gv,0],cmv.new_point[gv,1]]

                    # CMV in 3d cartesian coordinates

                    # cartesian initial point
                    r0 = 1
                    x0 = r0 * np.sin(prevzenith) * np.cos(prevazimuth)
                    y0 = r0 * np.sin(prevzenith) * np.sin(prevazimuth)
                    z0 = r0 * np.cos(prevzenith)

                    # cartesian terminal point
                    r1 = z0 / np.cos(currzenith)
                    x1 = r1 * np.sin(currzenith) * np.cos(currazimuth)
                    y1 = r1 * np.sin(currzenith) * np.sin(currazimuth)
                    z1 = r1 * np.cos(currzenith)

                    # CMV increment
                    dx = x1 - x0
                    dy = y1 - y0
                else:
                    dx = 0
                    dy = 0

                dt = cmv.lastimg

                # grid image vector points
                if cmv.npoints > 0:

                    x = cmv.old_point[:,1]; y = cmv.old_point[:,0]
                    cmv.old_point_grid = cam.mapping_backward(x, y, ini.x_res,
                        ini.y_res, ini.grid_size, np.radians(ini.horizon), cbh=cld.cbh,
                        saz=saz, sza=sza, cbh_flag = ini.cbh_flag, shadow_flag = ini.shadow_flag)

                    x = cmv.new_point[:,1]; y = cmv.new_point[:,0]
                    cmv.new_point_grid = cam.mapping_backward(x, y, ini.x_res,
                        ini.y_res, ini.grid_size, np.radians(ini.horizon), cbh=cld.cbh,
                        saz=saz, sza=sza, cbh_flag = ini.cbh_flag, shadow_flag = ini.shadow_flag)

                else:

                    cmv.old_point_grid = np.nan
                    cmv.new_point_grid = np.nan

                # Calculate CMV from optical flow tracked points
                # get u/v components in pixel/s
                cmv.calcWind(ini)

                # check quality against last vectors ( consistency check ) and calculate average CMV
                cmv.checkConsistency(t2target=ini.cbh_flag)

                # calculate a temporal smoothed CMV
                cmv.smoothWind()

                cmv.theta, cmv.azimuth, cmv.x, cmv.y = camera_cmv.predict(x1, y1,
                    z1, dx, dy, dt, cam, ini.fcst_horizon)
                cmv.flag = np.array(cmv.flag)

                print("Cloud Motion...\t\tfinished in %.1f seconds\t%d raw, %d good, direction: %.1f speed: %.1f" % \
                        (time.time() - st,cmv.npoints,np.sum(cmv.flag),np.degrees(cmv.mean_direction), \
                            cmv.mean_speed) )

            #-----------------------------------------------------------------------


            #=======================================================================
            # Cloud Base height calculations
            #=======================================================================

            # not yet implemented

            #-----------------------------------------------------------------------

            st = time.time()

            # project binary cloud map to the surface grid
            cmap = cam.grid(binary_image,map_x,map_y).astype(np.float32)
            rbr_grid = cam.grid(img.rbr,map_x,map_y).astype(np.float32)
            gray_grid = cam.grid(img.orig_gray,map_x,map_y).astype(np.uint8)
            orig_grid = cam.grid(img.orig_color, map_x,map_y).astype(np.uint8)

            # convert cloud map to 8-bit unsigned integer (0 = cloud & mask; 255 = clear sky)
            cmap[cmap==1]=256   # clear sky
            cmap[cmap==0]=0     # cloud & mask
            cmap[(cmap>1)&(cmap<=255)]=1 # transistions
            cmap[cmap==256]=255 # clear sky

            #=======================================================================
            # Smooth shadow edges
            #=======================================================================
            if ini.edge_smoothing:
                cmap[cmap==0] = np.nan
                if ini.cbh_flag:
                    # cloud edge ( sunspot opening angle )
                    sunopen = np.radians(0.8)
                    # sun diameter in meter
                    d_sun = cld.cbh * ( np.tan(sza+sunopen) - np.tan(sza-sunopen))
                    # kernel size ( sun diameter in grid cells )
                    kernel = np.int(d_sun / ini.x_res) # kernel size
                    if (kernel % 2) == 0: kernel += 1 # kernel must be odd
                else:
                    kernel = ini.median_filter - 2
                cmap = cv2.GaussianBlur(cmap,(kernel,kernel),0).astype(np.float32)
            else:
                cmap = cmap.astype(np.float32)





            #=======================================================================
            # Irradiance Forecast
            #=======================================================================
            horizon = int(ini.fcst_horizon / ini.fcst_res)
            if ini.flow_flag and ini.fcst_flag:

                # Initialise datasets
                for i in range(0, nstations):
                    pyr[i].reset_fcst_arrs(ini)

                if ini.cbh_flag:
                    # shadow length in meters
                    shad_length = float(cld.cbh) * np.tan(sza)
                    # shadow length in grid cells in both directions
                    u_shad0, v_shad0 = misc.latlon2grid(ini.y_res, ini.x_res, shad_length, saz)
                else:
                    u_shad0, v_shad0  = 0, 0

                # convert cmv speed in m/s to cmv speed in grid cells / s
                if np.degrees(cmv.std_direction) <= 30:
                    ucell, vcell = misc.latlon2grid(ini.x_res,ini.y_res,
                                                    cmv.mean_speed, cmv.mean_direction)
                else:
                    ucell = np.nan; vcell = np.nan

                csza = csk.cossza[csk.tind:csk.tind + ini.fcst_horizon]
                cghi_fcst = csk.ghi[csk.tind:csk.tind + ini.fcst_horizon]
                cdni_fcst = csk.dni[csk.tind:csk.tind + ini.fcst_horizon]

                sflag = True

                for i in range(0, ini.fcst_horizon):

                    # clearsky value for this lead time
                    cskval = csk.ghi[csk.tind + i]

                    # calculate a correction for changes in sunposition
                    dt = [ini.actdate + timedelta(seconds=+i)]
                    sun_fcst = solar.compute( dt , ini.lat0, ini.lon0, method="spencer" )
                    if ini.cbh_flag:
                        rfac = cld.cbh * np.tan(sun_fcst['zenith'][0])
                    else:
                        rfac = 0

                    ushadow, vshadow = misc.latlon2grid( ini.y_res, ini.x_res,
                                                        rfac, sun_fcst['azimuth'] )

                    sh_diff_u = ushadow - u_shad0
                    sh_diff_v = vshadow - v_shad0

                    # if no CMV could be detected, do not apply any cloud movement
                    if np.isnan(ucell): ucell = 0
                    if np.isnan(vcell): vcell = 0

                    # cloud advection + sun ray tracing
                    u = int(np.round(float(i) * ucell + sh_diff_u,0))
                    v = int(np.round(float(i) * vcell + sh_diff_v,0))

                    # x,y image coordinates of sun in original image without rotation
                    xsun_fcst, ysun_fcst = cam.sphere2img(sun_fcst['zenith'][0], \
                        sun_fcst['azimuth'][0], rotate=True)

                    if not ini.cbh_flag:
                        if ucell > 0:
                            xs = xsun_fcst
                            ys = ysun_fcst
                        else:
                            xs = xsun
                            ys = ysun
                        # position of camera/sunspot in gridded image
                        sun_grid = cam.mapping_backward(xs, ys,
                                ini.x_res, ini.y_res, ini.grid_size, np.radians(ini.horizon),
                                cbh_flag = False, shadow_flag = ini.shadow_flag)
                        pyr[0].y0, pyr[0].x0 = sun_grid[0]
                    else:
                        for t in range(0, nstations):
                            pyr[t].y0, pyr[t].x0 = pyr[t].map_y, pyr[t].map_x


                    # calculate cloud grid for each time step
                    for t in range(0, nstations):
                        try:
                            pyr[t].fpos[i] = ([pyr[t].x0 + v, pyr[t].y0 - u])
                        except ValueError:
                            continue

                    try:
                        rbr_val = rbr_grid[int(pyr[0].fpos[i][0]),int(pyr[0].fpos[i][1])]
                        gray_val = gray_grid[int(pyr[0].fpos[i][0]),int(pyr[0].fpos[i][1])]
                    except IndexError: # out of image
                        rbr_val = -1
                        gray_val = -1

                    if ini.csa_correction:
                         # Circumsolar area correction
                        if (rbr_val > ini.rbr_thres-0.02) & (gray_val > (0.7*255)) & sflag:
                            pyr[0].fdhi[i] = pyr[0].csi_min * cskval
                            pyr[0].fdni[i] = pyr[0].dni_csi_now * cdni_fcst[i]
                            pyr[0].fghi[i] = pyr[0].fdhi[i] + pyr[0].fdni[i] * csza[i]
                            pyr[0].bin[i] = -1
                            pyr[0].ftime[i] = i
                        else:
                            # Outside of circumsolar area
                            forecast.stations(cmap, nstations, pyr, ini, cskval, csza[i], i)
                            sflag = False
                    else:
                        # Binary mapping
                        forecast.stations(cmap, nstations, pyr, ini, cskval, csza[i], i)

                print("Forecast data...\tfinished in %.1f seconds" % round(time.time() - st,1))
                st = time.time()

                # Get reprojected fisheye pixel coordinates of forecast path for drawing purpose
                if ini.draw_forecast_path:
                    k = ini.statlist[0]
                    pyr[k].fpos = np.array(pyr[k].fpos,dtype=np.int)
                    # Get image pixel coordinates of forecast path
                    posis = (pyr[k].fpos[:,0]>=0) & (pyr[k].fpos[:,1]>=0) & \
                        (pyr[k].fpos[:,0] < ini.grid_size) & \
                        (pyr[k].fpos[:,1] < ini.grid_size)
                    pyr[k].origx = np.array([ int(round(map_y[xs,ys],0)) \
                        if not np.isnan(map_y[xs,ys]) else -1 \
                        for xs,ys in pyr[k].fpos[posis,:] ])
                    pyr[k].origy = np.array([ int(round(map_x[xs,ys],0)) \
                        if not np.isnan(map_x[xs,ys]) else -1 \
                        for xs,ys in pyr[k].fpos[posis,:] ])

                bias_period = ini.bias_period
                n_steps = int(bias_period / ini.rate) # number of last analysis values for bias correction
                n_meas = int(bias_period) # number of last measurement values for bias correction
                for t in range(0, nstations):
                    pyr[t].aghi[0:-1] = pyr[t].aghi[1:].copy()
                    pyr[t].aghi[-1] = pyr[t].fghi[0]
                    pyr[t].aghi_time[0:-1] = pyr[t].aghi_time[1:].copy(); pyr[t].aghi_time[-1] = actnumdate
                    if not pyr[t].mflag: continue

                # Compute final bias
                if ini.radiation:
                    ana_mbe = np.nanmean(pyr[0].aghi[-n_steps:] - pyr[0].ghi[pi-n_meas:pi])
                    meta['bias'] = ana_mbe
                    print("Bias correction...\tfinished in %.1f seconds\tBias GHI = %.2f" % (time.time() - st, ana_mbe))

            #------------------------------------------------------------------------------


            st = time.time()

            # Determine measurements and clear sky irradiance for the current forecast horizon
            if csk.tind > 0:
                fcsk = csk.ghi[csk.tind:csk.tind+horizon]
            else:
                fcsk = np.repeat(np.nan, horizon)
            for t in range(0, nstations):
                if pyr[t].tind > 0 and pyr[t].mflag:
                    pyr[t].fmeas = pyr[t].ghi[pyr[t].tind:pyr[t].tind+ini.fcst_horizon]
                else:
                    pyr[t].fmeas = np.repeat(np.nan, horizon)

            #===================================================================
            # Forecast averaging
            #===================================================================
            for t in range(0, nstations):
                pyr[t].ftime = actnumdate + np.array(pyr[t].ftime[::ini.fcst_res])[1:]
            if ini.fcst_res > 1 and ini.fcst_flag:
                for t in range(0, nstations):
                    pyr[t].fghi = np.nanmean(np.split(pyr[t].fghi,ini.fcst_horizon/ini.fcst_res),axis=1)
                    pyr[t].fdni = np.nanmean(np.split(pyr[t].fdni,ini.fcst_horizon/ini.fcst_res),axis=1)
                    pyr[t].fdhi = np.nanmean(np.split(pyr[t].fdhi,ini.fcst_horizon/ini.fcst_res),axis=1)
                    if pyr[t].tind > 0 and pyr[t].mflag and not ini.live:
                        pyr[t].fmeas = np.nanmean(np.split(pyr[t].fmeas,ini.fcst_horizon/ini.fcst_res),axis=1)
                    fcsk = np.nanmean(np.split(fcsk,ini.fcst_horizon/ini.fcst_res),axis=1)

            #===================================================================
            # Visualization
            #===================================================================

            plt_mod = (cnt-1) % ini.plot_int
            plot_any = np.any([ini.plot_standard, ini.plot_detection, \
                ini.plot_cmv, ini.plot_origin])
            if plt_mod == 0 and plot_any:

                st = time.time()

                # PLot equilines of incidence Angle
                if ini.plot_incidence_angle:
                    drawings.incidence_angle(img.orig_color_draw, cam)
                # Plot equilines of distance
                if ini.plot_equi_distance:
                    drawings.equi_distance(img.orig_color_draw, cam)
                # Plot equilines of circumsolar angular distance
                if ini.plot_circumsolar_area:
                    drawings.circumsolar_area(img.orig_color_draw, cam.theta,
                                              cam.phi2, sza, saz)
                # Plot the daily sunpath
                if ini.plot_sunpath:
                    xsolar, ysolar = cam.sphere2img(sun['zenith'], sun['azimuth'], rotate=True)
                    drawings.sunpath(img.orig_color_draw, actdate, ini, cam)
                # Plot the forecast path
                if ini.draw_forecast_path and ini.fcst_flag:
                    k = ini.statlist[0]
                    drawings.forecast_path(img.orig_color_draw, pyr[k].origx, pyr[k].origy)
                    drawings.forecast_path(img.binary_color, pyr[k].origx, pyr[k].origy)
                # Draw CMV in image
                if ini.flow_flag and ini.draw_cmv:
                    cmv_mask = optical_flow.drawVectors(cmv.flag,cmv.old_point,\
                        cmv.new_point, cmv_mask, vcolor=[0,0,255], \
                        vsize=5,skip=2,mask_flag=True)
                    cmv.cmv_mask = np.nansum(cmv_mask[:,:,:],axis=2)>0

                if ini.mode == 0 and ini.radiation:
                    # Shadow map (gray color)
                    draw_field = np.uint8(cmap)
                    draw_field[np.isnan(cmap)] = 0 # mask
                    draw_field[cmap==0] = 0 # mask
                    draw_field[cmap==1] = 100 # cloud
                else:
                    draw_field = cam.grid(binary_image, map_x, map_y).astype(np.float32)
                    draw_field[draw_field==0] = np.nan
                    draw_field[draw_field==1] = 0





                # Radiation values
                ghi = np.nanmean(ghi); dhi = np.nanmean(dhi); dni = np.nanmean(dni)

                # Plot parameters
                params = dict( sza = np.degrees(sza), saz = np.degrees(saz), dni = dni, \
                    dhi = dhi, ghi = ghi, txt=radtxt, csi_ghi = csi_ghi, \
                    csi_dhi = csi_dhi, csi_dni = csi_dni, circ = circ, gray = gray , \
                    cbh = cld.cbh, cc = cc_img, imgclass = cloud_class, \
                    imgprob = cloud_class_prob,  mdni = pyr[0].dni_cam / csk.dni[csk.tind], \
                    gmin = pyr[0].csi_min, img_qc=meta['img_qc'])


                # Standard output
                if ini.plot_standard:
                    if ini.live:
                        outfile = ini.outdir + os.sep + 'tmp' + os.sep + 'si_current.' + ini.outformat
                    else:
                        if ini.mode == 0:
                            outpath = ini.outdir + os.sep + 'plots' + os.sep + ini.locdate_str + os.sep + 'standard'
                        if ini.mode == 1:
                            outpath = ini.outdir + os.sep + 'plots' + os.sep  + 'standard'
                        if not os.path.exists(outpath): os.makedirs(outpath)
                        outfile = outpath + os.sep + img.datetimestr + '.' + ini.outformat
                    visualization.plot(outfile,\
                        img, actdate, nstations, pyr, csk, ini, cmv, xsun,
                        ysun, mask_bool, csltype, draw_field,
                        features, hist_flag=ini.plot_histo, params=params)


                # Cloud segementation
                if ini.plot_detection:
                    if ini.mode == 0:
                        outpath = ini.outdir + os.sep + 'plots' + os.sep + ini.locdate_str + os.sep + 'segmentation'
                    elif ini.mode == 1:
                        outpath = ini.outdir + os.sep + 'plots' + os.sep + 'segmentation'
                    if not os.path.exists(outpath): os.makedirs(outpath)
                    outfile = outpath + os.sep + img.datetimestr + '.' + ini.outformat
                    visualization.plot_detection_full(outfile, ini, img, mask, **params)
                if ini.plot_origin:
                    if ini.mode == 0:
                        outpath = ini.outdir + os.sep + 'plots' + os.sep + ini.locdate_str + os.sep + 'origin'
                    elif ini.mode == 1:
                        outpath = ini.outdir + os.sep + 'plots' + os.sep + 'origin'
                    if not os.path.exists(outpath): os.makedirs(outpath)
                    outfile = outpath + os.sep + img.datetimestr + '.' + ini.outformat
                    visualization.plot_original_mod(img, outfile, mask_bool | img_mask,
                                actdate, ini, cam, **params)

                if ini.plot_cmv:
                    if ini.mode == 0:
                        outpath = ini.outdir + os.sep + 'plots' + os.sep + ini.locdate_str + os.sep + 'cmv'
                    elif ini.mode == 1:
                        outpath = ini.outdir + os.sep + 'plots' + os.sep + os.sep + 'cmv'
                    if not os.path.exists(outpath): os.makedirs(outpath)
                    outfile = outpath + os.sep + img.datetimestr + '.' + ini.outformat
                    # Compute mapping matrices for dimensionless perspective projection
                    tmp_x, tmp_y = cam.backward(ini, cld.cbh, saz, sza, \
                               cbh_flag=False, limit=ini.horizon, \
                                shadow_flag=False)
                    # Plot the CMV results
                    visualization.plot_cmv(outfile, img, ini, cam, cmv, xsun, ysun, mask, tmp_x, tmp_y, **params)
                    tmp = 0; tmp_x = 0; tmp_y = 0; del tmp; gc.collect()

                # Copy image to archive
                if ini.live and ini.archive_img and plt_mod == 0:
                    shutil.copy(outfile,ini.outdir + os.sep + 'plots' + os.sep + \
                    ini.datestr + os.sep + img.datetimestr + '.' + ini.outformat)

                print("Drawing gfx...\t\tfinished in %.1f seconds" % round(time.time() - st,1))



            #===================================================================
            # Data output
            #===================================================================
            write_flag = (cnt-1) % ini.write_interval == 0
            if write_flag:
                horizon = int(ini.fcst_horizon / ini.fcst_res)
                if (ini.append or cnt > 2):
                    append = True
                else:
                    append = False


                st = time.time()
                if ini.write_meta:

                    if ini.live:
                        output.write2Table(ini.outdir + os.sep + 'tmp' + os.sep + 'current_meta.dat',init=True,**meta)

                    if (ini.append or cnt > 2) and (os.path.exists(ini.outfile)):
                        output.write2Table(ini.outfile,init=False,**meta)
                    else:
                        output.write2Table(ini.outfile,init=True,**meta)

                if ini.write_hdf5:

                    data = {}
                    data['version'] = __version__

                    if ini.fcst_flag:
                        data['fcst'] = pyr
                    data['csk'] = fcsk
                    data['meta'] = meta
                    if ini.features:
                        data['meta_rad'] = meta_rad
                    if ini.features:
                        data['features'] = features
                    if ini.flow_flag:
                        data['cmv'] = cmv
                    if ini.radiation:
                        if not ini.live:
                            data['ghi'] = pyr[0].ghi[pi]
                            if len(pyr[0].dhi) > 0:
                                data['dhi'] = pyr[0].dhi[pi]
                                data['dni'] = pyr[0].dni[pi]
                        data['cls_dni'] = csk.dni[csk.tind]
                        data['cls_dhi'] = csk.dhi[csk.tind]
                        data['cls_ghi'] = csk.ghi[csk.tind]

                    if not os.path.exists(ini.hdffile) or \
                            (not ini.append and (cnt-1) == ini.write_interval):
                        hdfinit = True
                    else:
                        hdfinit = False
                    output.write2HDF5(ini.hdffile, actnumdate, ini,
                        init=hdfinit, **data)

                    del data
                    data = []
                    gc.collect()

                if ini.write_cmv:
                    # Write motion vector position for cloud height estimations
                    outpath = ini.outdir + os.sep + 'cmv'
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                    filename = outpath + os.sep + datetime.strftime(actdate,'%Y%m%d.hdf5')
                    output.write_cmv(filename, actnumdate, cmv, append=append)


                print("Write data...\t\tfinished in %.1f seconds" % (time.time() - st))


            orig_gray_old = img.orig_gray.copy()
            orig_color_old = img.orig_color.copy()
            rbr_old = img.rbr.copy()
            filename = img.filename

            # Now update the previous frame and previous points for optical flow
            if ini.flow_flag and np.sum(cmv.flag) > 5:
                counter += 1
                flow.p0 = cmv.new_point.reshape(-1,1,2)
            elif np.sum(cmv.flag) <= 5: # Reset counter if no valid CMV are found
                counter = 1000

            # clean memory
            del img, cmap
            gc.collect()


        else:
            print("Skip this image")


        print("Finished in %.1f seconds\n" % (time.time() - start_time))
        if ini.live: cmv.lastimg = round(time.time() - start_time,1)

    if ini.live_stream: cap.release()
    print(" ")
    print("Program finished in %.1f seconds" % (time.time() - startstart_time))

    if ini.log_flag:
        sys.stdout.close()
        sys.stderr.close()
