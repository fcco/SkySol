from datetime import datetime
import numpy as np
import os

class ini:

    def __init__(self,indate=""):

        #-----------------------------------------------------------------------
        # Location specific settings
        #-----------------------------------------------------------------------
        self.lat0 = 12.956785    # camera latitude
        self.lon0 = 80.2142      # camera longitude
        self.tz = 'UTC'          # Timezone for image filenames

        # Flags for different modi ( False = switch off, True = switch on )
        self.log_flag = False # Standard out should be written to log file
        self.radiation = False  # Do not use irradiance data. No forecasts. Image processing only.
        self.archive_img = True # Only in live mode: Archive output plots

        #---------------------------------------------------------------------------
        # Image Processing
        #---------------------------------------------------------------------------
        self.median_filter = 9        # pixel size of median filter (<10, odd-numbered)
        self.edge_smoothing = True    # apply gaussian filter to shadow/irradiance map to smooth cloud edges
        self.quality_check = False    # Not implemented yet correctly

        #-----------------------------------------------------------------------
        # Cloud detection
        #-----------------------------------------------------------------------
        self.rbr_thres = 0.82            # Threshold for Red-Blue-Ratio
        self.csl_flag = True            # Use Clear Sky Library for RBR adjustment
        self.csl_weighting = 0.40         # Clear sky reference weighting
        self.clear_correction = 0.0008   # CSL correction factor

        # Image Projection
        self.cbh_flag = True         # Get and use cloud base height
        self.shadow_flag = True      # Compute a cloud shadow projection

        #---------------------------------------------------------------------------
        # Grid settings
        #--------------------------------------------------------------------------
        self.x_res = 10        # east-west resolution in meter
        self.y_res = 10        # north-south resolution in meter
        self.grid_size = 1000  # grid_size in cells

        #---------------------------------------------------------------------------
        # Forecast settings
        #---------------------------------------------------------------------------
        # Switch on/off forecasting
        self.fcst_flag = True
        # Forecast horizon in seconds
        self.fcst_horizon = 900
        # Forecast output resolution in seconds (60=1min), internal res. is always 1s)
        self.fcst_res = 20
        # Forecast locations (specifiy list of lat/lon pairs)
        self.flocs = [[12.96,80.2],[12.947222, 80.211269]]
        # A list of station numbers for timeseries plotting (up to now only one station number can be used)
        self.statlist = [1]
        # Bias correction using latest measurements
        self.bias_correction = False
        # Circumsolar area correction for single point forecast
        self.csa_correction = True
        # number of images that should be skipped for processing (1 = every image is used)
        self.camera_res = 1

        #---------------------------------------------------------------------------
        # Optical Flow settings
        #---------------------------------------------------------------------------
        self.flow_flag = True       # Switch on/off optical flow
        self.renew_cmv = 30         # Time intervall [s] for new corner detection

        #---------------------------------------------------------------------------
        # Radiation model
        #---------------------------------------------------------------------------
        self.csi_mode = "hist"     # Mode to use ("fix", "hist", "meas")
        self.avg_csiminmax = 1800  # for "hist" (use past x measurements)
        self.hist_bins = 100       # for "hist" (number of bins used for find peaks)

        #---------------------------------------------------------------------------
        # Image features
        #---------------------------------------------------------------------------
        self.features = False                    # Switch for feature computation. Must be On for cloud classification
        self.contour_flag = False                # Additional contour features

        #---------------------------------------------------------------------------
        # Image Classification
        #---------------------------------------------------------------------------
        self.cloud_class_learn = False          # Train cloud classification model
        self.cloud_class_apply = False          # Apply cloud classification model
        self.rank_features = False              # Rank feature importance to select most important features
        self.nFeatures = 10                     # Number of ranked features to select
        self.print_features = True              # Print feature result output
        self.class_model = "SVC"                # ML model to use


        #---------------------------------------------------------------------------
        # Text/File output
        #---------------------------------------------------------------------------
        self.write_meta = True         # Write a file with meta information for each image
        self.write_hdf5 = True         # Standard Output
        self.write_csl = False          # Save image in clear sky library
        self.write_cmv = False         # Write positions of cloud motion vectors
        self.append = False             # if True, append to existing files, else overwrite existing files

        #---------------------------------------------------------------------------
        # Some default settings, can be changed
        #---------------------------------------------------------------------------
        self.def_cbh = 1500          # Default Cloud base height. Used if no other information available
        self.min_sun_elev = 2        # Lower threshold for sun elevation to consider image (> horizon)
        self.horizon = 70            # the horizon (zenithal angle) that is excluded in images for the analysis
        self.avg_csi = 60            # Period of time [s] in the past for averaging CSI and CloudClass
        self.write_interval = 1      # interval of data output, 1 means every image
        self.cmv_temp_avg = 10
        self.bias_period = 10
        self.cbh_source = ""
        self.sunspot = 3             # Size of sunspot in degrees for masking the sun

        # Graphic output settings
        self.plot_standard = True                       # Standard forecast plot
        self.plot_detection = True                      # Cloud detection analyses
        self.plot_origin = False                         # Single original image with calibration results
        self.plot_cmv = True                          # Results of cloud motion

        # Additional drawings
        self.plot_histo = True                          # Plot the RBR histogram
        self.plot_incidence_angle = False               # Draw isolines of pixel incidence angles
        self.plot_equi_distance = False                 # Draw isolines of distance to camera
        self.plot_circumsolar_area = False              # Draw isolines of sun-pixel-angle (SPA)
        self.plot_sunpath = False                       # Draw daily sunpath
        self.draw_forecast_path = True                  # Draw forecast path
        self.draw_cmv = True                            # Draw cloud motion vectors
        self.outformat = 'png'                          # Image output format
        self.plot_int = 3                               # Interval for plotting output
        self.plot_features = False                      # Write feature values to plot
        self.plot_last_vals = 300                       # Standard output - timeseries: Number of past measurements
        self.draw_contours = False

        #---------------------------------------------------------------------------
        # Camera settings
        #---------------------------------------------------------------------------
        self.datetimefmt="%Y%m%d_%H%M%S"                # Date format used for image filenames
        self.stream_flag = False                        # Live images from Webcam stream
        self.http = 'http://134.106.241.200/video.mjpg' # Webcam Stream URL
        self.imgformat = 'jpg'                          # Camrea image format
        self.scale = 1               # scaling factor to shrink images
        self.rate =  10              # Image acquisition rate in [s]
        self.cy = 948. * self.scale # center row
        self.cx = 964. * self.scale # center column
        self.cam_c = 1.001451 # affine parameters "c","d","e"
        self.cam_d = 0.000284
        self.cam_e = -0.000164
        self.fx = 960. * self.scale
        self.fy = 960. * self.scale
        self.imgsize = [1920 * self.scale , 1920 * self.scale ]    # Image size in pixel
        self.radius = 960. * self.scale
        # polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world
        self.polyc = [ -6.797272e+02, 0.000000e+00, 6.431645e-04, -3.918755e-07, 4.276154e-10 ]
        # polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam
        self.invpoly = [1000.447606, 548.778375, -34.850836, 87.609269, 38.008847, -13.102088, 35.947334, 29.217085, -20.384776, -22.185998, -4.989513 ]
        # rotation angles
        self.rot_angles = [0, 0, np.radians(33)]
        # flip image - 0: horizontal, 1: vertical, -1: both; Finally, E is right; S is bottom; W is left
        self.flipping = 1
        #---------------------------------------------------------------------------
        # Directory names
        #---------------------------------------------------------------------------

        # root directory / workspace
        self.rootdir = '/home/l1nx/NIWE'
        # sky images archive directory (assumes images to be saved in YYYYmmdd subdirectories)
        self.picpath = '/media/l1nx/A54F-A5EE'
        # measurement data file
        mfmt = 'Data_%Y%m%d.csv'
        mdata = "X:\" + os.sep + 'data'
        self.data_fmt = mdata + os.sep + mfmt
        # Clear Sky Library
        self.cslfile = self.rootdir + os.sep + 'data' + os.sep + 'CSL.h5'
        # image mask file
        self.maskfile = self.rootdir + os.sep + 'config' + os.sep + 'mask_priliminary.jpg'

        # Data output directory
        self.outdir = self.rootdir + os.sep + 'skysol'             # The directory for several output
        self.outdir_live = self.outdir
        nowstr = datetime.strftime(datetime.utcnow(),"%Y%m%d_%H%M%S")

        self.datestr = datetime.strftime(indate, "%Y%m%d")
        # Logfile
        self.logfile = self.outdir + os.sep + 'log' + os.sep + nowstr + '_' + self.datestr + '.log'
        # Meta data file
        self.outfile = self.outdir + os.sep + 'meta' + os.sep + self.datestr + '.meta'
        # HDF5 file with forecast results
        self.hdffile = self.outdir + os.sep + 'hdf5' + os.sep  + self.datestr + '.hdf5'
        # cloud classification models
        self.cloud_class_model = self.rootdir + os.sep + 'classification' + os.sep + 'cloud_class.h5'


        if indate != "":
            self.year = indate.year
            self.month = indate.month
            self.day = indate.day
            self.hour = indate.hour
            self.minute = indate.minute
            self.actdate = indate
