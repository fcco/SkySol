#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
title	       : SkySol Validation tool
description	   : Sky Imager based Solar Irradiance Analyses and Forecast Validation Tool
author		   : Thomas Schmidt
date	       : 20141218
last change    : 20181228
version		   : 0.5
usage		   : python3 validate.py -d YYYYmmdd
notes		   :
python_version : 3.5

This validation tool evaluates SkySol output. It performs a few visualizations
of forecast vs. measurement data. It is necessary to provide at least GHI measurements
either extracted from the SkySol output file (if given) or by reading in external
measurement data file.

Functions for computing the error metrics are provided in validation/misc.py
Functions for drawing the plots are provided in validation/drawings.py
==============================================================================
"""
__author__ = "Thomas Schmidt"
__copyright__ = "Copyright 2014, University of Oldenburg"
__version__ = "0.5"
__maintainer__ = "Thomas Schmidt"
__email__ = "t.schmidt@uni-oldenburg.de"
__status__ = "development"

import sys
import os
import warnings
from datetime import datetime, timedelta
import pytz
import time
import numpy as np
import gc
from skysol.validation import misc
from skysol.validation import drawings

import matplotlib as mpl
mpl.use('agg')
from matplotlib.pyplot import *

cfg = {}
cfg['Paths'] = {}
cfg['Paths']['HDF5_PATH'] = "/home/l1nx/NIWE/skysol/hdf5"
cfg['Paths']['Meas_PATH'] = ""

cfg['Misc'] = {}
# station number
cfg['Misc']['station'] = 0
# forecast horizon in seconds
cfg['Misc']['horizon'] = 900
# forecast resolution in seconds
cfg['Misc']['step'] = 20


if __name__ == "__main__":

    processstart = time.time()

    warnings.simplefilter("ignore", RuntimeWarning)

    adir = os.path.dirname(sys.argv[0])
    if adir == "": adir = "."
    #cfg = misc.loadConfig(adir + '/skysol/validation')
    CFG_PATH = "./config"
    args = misc.argHandler(sys.argv[0:],cfg)

    # import user and location specific settings
    from skysol.settings import settings

    ini = settings.ini(indate=args['date'])
    lat, lon = ini.lat0, ini.lon0
    if args['outpath']:
        figpath = args['outpath']
    else:
        figpath = ini.outdir + os.sep + 'eval' + os.sep + ini.datestr
    os.makedirs(figpath, exist_ok=True)

    tz_flag = ini.tz


    fmt = 'png' # output format for graphics
    color_flag = True # True if graphics should be in colour ( else grayscale )
    quiet = args['quiet']
    horizon = args['horizon']
    step = args['step']
    station = args['station']
    fcn = "forecast"
    hor_limit = 75
    acc_flag = True
    interactive = False
    horizon_from_file = False

    #-------------------------------------------------------------------------------
    # Input skyimager forecast file
    #-------------------------------------------------------------------------------
    if not quiet: print("Read data...")
    exclude=['forecast*','rgb','azimuth']

    # Read in forecast data
    data = misc.read_hdf5(args, inpath=args['inpath'], fmt="%Y%m%d.hdf5", \
        exclude=exclude, horizon=horizon, timekey="time")
    if horizon_from_file: horizon = data[fcn+'/fghi'].shape[1]

    #-------------------------------------------------------------------------------
    # Measurement data
    #-------------------------------------------------------------------------------
    meas = misc.read_niwe(datetime.strftime(ini.actdate, ini.data_fmt))
    try:
        m_time = np.int64(meas['time'][:])
    except KeyError:
        m_time = np.int64(meas['datetime'][:])

    f_time = np.int64(data['time'][:])
    #unique = list_duplicates(f_time)
    #-------------------------------------------------------------------------------
    # Merge data files ( intersection )
    #-------------------------------------------------------------------------------
    find = np.in1d(f_time,m_time) #& unique   # Forecast instances
    mind = np.in1d(m_time,f_time[find])       # Measurement instances in Forecast

    m_time = m_time[mind]
    f_time = f_time[find]

    utcdates = np.array([ datetime.utcfromtimestamp(ts) for ts in f_time ])

    #-------------------------------------------------------------------------------
    # reduced subset of relevant data needed for validation
    #-------------------------------------------------------------------------------
    try:
        # Read measurements from SkySol file
        mghi = np.float16(data[fcn+'/mghi'][find,:horizon,station])
        archive = True
    except (IndexError, KeyError):
        # Read measurements from measurement archive
        archive = False
        tghi = meas['ghi'][:]
        mghi = np.empty((len(tghi),int(horizon/step)),dtype=np.float16)
        for i in range(0,horizon,step):
            mghi[:,int(i/step)] = np.roll(tghi,-i)
        mghi = mghi[mind,:]
    fghi = data[fcn+'/fghi'][find,:horizon,station]
    cghi = data[fcn+'/cghi'][find,:horizon]
    mghi = data[fcn+'/mghi'][find,:horizon,station]

    # Cloud motion vectors
    try:
        cmv = {}
        cmv['time'] = data['time']
        cmv['cmv_u'] = data['cmv/cmv_u'][find]
        cmv['cmv_v'] = data['cmv/cmv_v'][find]
        cmv['cmv_spd_sigma'] = data['cmv/cmv_spd_sigma'][find]
        cmv['cmv_dir_sigma'] = data['cmv/cmv_dir_sigma'][find]
    except:
        pass

    # Image zenithal angle of predicted pixel
    try:
        fsza = data[fcn+'/zenith'][find,:,station]
    except:
        pass

    # Cloud classification data
    try:
        cld_flag = True
        cldclasses = data['meta/cloudclass'][find]
    except KeyError:
        cld_flag = False
        pass

    # Cloud cover data
    try:
        cc_flag = True
        cc = data['meta/cloud_cover'][find]
        cc_rect = data['meta/cloud_cover_stretched'][find]
    except (IndexError, KeyError):
        cc_flag = False

    del data
    del meas

    gc.collect()

    #-------------------------------------------------------------------------------
    # Solar geometry and clearsky calculations
    #-------------------------------------------------------------------------------
    # if not quiet: print("Solar geometry and clear sky calculations...")
    #
    from skysol.lib import clearsky, solar
    #
    # sundates = np.array([ utcdates[0] + timedelta(seconds=i) \
    #     for i in range(0,int((utcdates[-1]-utcdates[0]).total_seconds()+horizon)) ])
    #
    # # Compute solar position for whole day
    sun = solar.compute(utcdates, lat, lon, method="spencer")
    coszen = np.cos(sun['zenith'][:])
    #
    # # Compute and get clear sky reference for current day
    # csk = clearsky.clearsky(utcdates, lat, lon, source="pvlib")
    # cglobal = csk.ghi
    # cdiffuse = csk.dhi
    #
    # elevation = 90-np.degrees(sun['zenith'])
    # all_ind = np.in1d(sundates,utcdates)
    #
    # cghi = np.empty((len(utcdates),int(horizon/step)),dtype=np.float16)
    # cossza = np.empty((len(utcdates),int(horizon/step)),dtype=np.float16)
    #
    # cdirect = (cglobal - cdiffuse) / coszen
    # print(cghi.shape, mghi.shape)
    # for i in range(0,horizon,step):
    #     cghi[:,int(i/step)] = np.roll(cglobal,-i)
    #     cossza[:,int(i/step)] = np.roll(coszen,-i)
    #
    # #cghi = cghi[all_ind,:]
    # #cossza = cossza[all_ind,:]
    #
    # #-------------------------------------------------------------------------------
    # Clear sky indizes and persistence reference forecast
    #-------------------------------------------------------------------------------
    if not quiet: print("Clear sky index and persistence...")

    # GHI
    csi = np.float16(np.divide(mghi[:,0],cghi[:,0]))
    csi_p = np.array([csi,]*horizon).T[:,::step]
    print(mghi.shape)
    pghi = np.float16(csi_p * cghi)

    # Remove Persistence values where no forecasts are available to make errors comparable
    nan_ind = ~np.isfinite(fghi[:,:]) | ~np.isfinite(pghi[:,:])
    pghi[nan_ind] = np.nan
    fghi[nan_ind] = np.nan
    mghi[nan_ind] = np.nan

    # Filter data
    si = np.degrees(sun['zenith']) < hor_limit

    if tz_flag == "UTC":
        dates = utcdates
        tlabel = 'Time [UTC]'
    else:
        tzi = pytz.timezone(tz_flag)
        dates = np.array([ pytz.UTC.localize(ts).astimezone(tzi) for ts in utcdates ])
        rcParams['timezone'] = tzi
        tlabel = 'Local Time'

    #-------------------------------------------------------------------------------
    # plot forecast evaluation
    #-------------------------------------------------------------------------------
    if not quiet:
        print("""
=========================================
\tGraphical output\t
=========================================""")
    dtstring = datetime.strftime(dates[si][0],"%Y%m%d")
    ststring = str(station)
    textkeys = dict(fontsize=14,horizontalalignment="left",backgroundcolor = "white", verticalalignment="top")

    # Measured GHI
    x = np.float32(mghi[si,:])
    # Forecast GHI
    y = np.float32(fghi[si,:])
    # Clear Sky GHI
    c = np.float32(cghi[si,:])
    # Persistence GHI
    p = np.float32(pghi[si,:])

    #--------------------------------------------------------------------------
    # Error vs. time ( x min lead )
    #--------------------------------------------------------------------------

    # specify the lead time to evaluate here (in seconds)
    lead = slice(120,int(300/step),None)
    drawings.plot_tseries_eval(dates[si], x, y, c, p, \
        lead = lead, slider=False, sampling="", averaging="")
    tight_layout()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_error_vs_time_2-5min.'+fmt, dpi=200); clf()

    #--------------------------------------------------------------------------
    # Error vs. lead time
    #--------------------------------------------------------------------------
    drawings.plot_lead_eval(dates[si], x, y, c, p, step=step)
    draw(); savefig(figpath + os.sep + dtstring + '_' + ststring + '_error_vs_lead_ghi.'+fmt, dpi=200); clf()


    #--------------------------------------------------------------------------
    # Scatter analysis plot for horizon 5min
    #--------------------------------------------------------------------------
    if not quiet: print("Scatter plots...")
    lead = int(300/step)
    f, ax1 = subplots(1, 1, figsize=(7,6))
    drawings.draw_scatter(ax1, x,y, y2=p, lead=lead,
                          plttitle="GHI forecast lead 5 min", param_name="($Wm^{-2}$)", \
                          interactive=False, ylab="Forecast", y2lab="Persistence")
    tight_layout()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_scatter.'+fmt, dpi=200); clf()

    if interactive:
        f, ax = subplots(1, 1, figsize=(10,8))
        drawings.draw_scatter(ax, x, y, y2=p,
                              plttitle="GHI forecast lead 5 min", param_name="($Wm^{-2}$)", \
                              interactive=True, ylab="Forecast", y2lab="Persistence")
        show()


    #--------------------------------------------------------------------------
    # Scatter analysis plot for horizon 5min (average)
    #--------------------------------------------------------------------------
    lead = int(300/step)
    f, ax1 = subplots(1, 1, figsize=(7,6))
    x = np.nanmean(np.float32(mghi[si,0:lead]),axis=1)
    y2 = np.nanmean(np.float32(pghi[si,0:lead]),axis=1)
    y =  np.nanmean(np.float32(fghi[si,0:lead]),axis=1)
    drawings.draw_scatter(ax1, x, y, y2=y2, interactive=False,
                          plttitle="GHI forecast 5 min average",
                          param_name="($Wm^{-2}$)",
                          ylab="Forecast", y2lab="Persistence")
    tight_layout()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_5avg_scatter.'+fmt, dpi=200); clf()
    lead = int(600/step)
    f, ax1 = subplots(1, 1, figsize=(7,6))
    x = np.nanmean(np.float32(mghi[si,0:lead]),axis=1)
    y2 = np.nanmean(np.float32(pghi[si,0:lead]),axis=1)
    y =  np.nanmean(np.float32(fghi[si,0:lead]),axis=1)
    drawings.draw_scatter(ax1,x,y,y2=y2, interactive=False,\
                plttitle="GHI forecast 10 min average", param_name="($Wm^{-2}$)",
                ylab="Forecast", y2lab="Persistence")
    tight_layout()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_10avg_scatter.'+fmt, dpi=200); clf()
    lead = int(900/step)
    f, ax1 = subplots(1, 1, figsize=(7,6))
    x = np.nanmean(np.float32(mghi[si,0:lead]),axis=1)
    y2 = np.nanmean(np.float32(pghi[si,0:lead]),axis=1)
    y = np.nanmean(np.float32(fghi[si,0:lead]),axis=1)
    drawings.draw_scatter(ax1,x,y,y2=y2, interactive=False, \
                plttitle="GHI forecast 15 min average", param_name="($Wm^{-2}$)",
                ylab="Forecast", y2lab="Persistence")
    tight_layout()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_15avg_scatter.'+fmt, dpi=200); clf()


    # Energy yield
    if not quiet: print("Energy yield...")
    f, ax = subplots(1, 1, figsize=(6,3))
    drawings.eyield_err(ax, np.float32(mghi[si,:]), np.float32(fghi[si,:]), \
         np.float32(pghi[si,:]), label1="Forecast", label2="Persistence", \
         step=step, color=color_flag)
    tight_layout()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_eyield.'+fmt, dpi=200); clf()


    #--------------------------------------------------------------------------
    # Mean absolute error (MAE) of GHI forecasts vs. viewing angle
    #--------------------------------------------------------------------------
    try:
        mae = np.float32(np.abs(np.subtract(fghi,mghi)))
        maep =  np.float32(np.abs(np.subtract(pghi,mghi)))
        if not quiet: print("MAE vs. Viewing angle...")
        f, ax = subplots(nrows=2, ncols=1, figsize=(18,9))
        drawings.plot_u_sza(ax, mae, maep, fsza)
        tight_layout()
        savefig(figpath + os.sep + dtstring + '_' + ststring + '_zenith_mae.'+fmt, dpi=100); clf()
    except:
        pass


    #--------------------------------------------------------------------------
    # Uncertainty vs Variability
    #--------------------------------------------------------------------------
    if not quiet: print("Uncertainty vs. variability...")

    ss = 12
    ws = 90
    n_iter = int(len(mghi[si,0]) / ss)
    cldcol = ["#9b59b6", "#2c6fbb", "#b04e0f", "#028f1e", '#A90016', "#34495e", "#ffad01"]
    cldnam = ['Cu','Ci & Cs','Cc & Ac','Clear','Sc','St & As','Cb & Ns']
    leads = (np.array([59,299,599,899])/step).astype(int)
    f, ax = subplots(2, 2, sharex='col', sharey='row', figsize=(12,5))
    for i in range(0,len(leads)):
        if i < 2: row = 0
        else: row = 1
        col = i % 2
        ts, Urr, Uprr, Vrr = misc.u_vs_v(m_time[si], mghi[si,:],
                                         fghi[si,:], pghi[si,:],
                                         cghi[si,:], step_size=ss,
                                         window_size=ws, horizon=leads[i])
        Urr = np.array(Urr); Vrr = np.array(Vrr); Uprr = np.array(Uprr)
        ct = []
        if cld_flag:
            cld = cldclasses[si]
            for j in ts:
                slc = slice(j*ss,(j*ss) + int(ws))
                a,b = np.unique(cld[slc],return_counts=True)
                try:
                    ct.append(a[np.where(b==np.max(b))[0][0]])
                except:
                    continue
        ax[row,col].set_title(str((leads[i]+1)*step)+'s')
        if len(Urr) == 0: continue
        drawings.plot_uv_scatter(ax[row,col], Urr, Uprr, Vrr, clclass=ct)

    sc = []
    for i in range(0,len(cldcol)):
        sc.append(scatter(0,0,c=cldcol[i],label=cldnam[i]))
    figlegend(handles=sc, labels=cldnam, loc="lower center",
           ncol=7, mode="expand", borderaxespad=0.)
    tight_layout()
    subplots_adjust(bottom=0.18)
    draw()
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_u_vs_v.' + fmt, dpi=200)
    clf(); close()


    #--------------------------------------------------------------------------
    # Cloud motion
    #--------------------------------------------------------------------------
    try:
        if not quiet: print("Cloud motion...")
        f, ax = subplots(nrows=2, ncols=1, sharex=True, figsize=(8,4))
        drawings.plot_cmv(ax[0], dates[si], cmv, par="Speed",\
            sampling="", plttitle="Cloud speed", slc=si)
        drawings.plot_cmv(ax[1], dates[si], cmv, par="Direction",\
            sampling="", plttitle="Cloud direction", xlabel=tlabel, slc=si)
        subplots_adjust(bottom=0.15, right=0.98)
        savefig(figpath + os.sep + dtstring + '_' + ststring + '_cmv.' + fmt, dpi=200)
        clf()
    except:
        pass

    #-------------------------------------------------------------------------------
    # Cloud cover
    #-------------------------------------------------------------------------------
    if cc_flag:
          if not quiet: print("Cloud cover...")
          f, ax = subplots(nrows=1, ncols=1, sharex=True, figsize=(6,4))
          drawings.plot_cloudcover_tseries(ax, dates[si], cc[si])
          subplots_adjust(bottom=0.15, right=0.98)
          savefig(figpath + os.sep + dtstring + '_' + ststring + '_cc.' + fmt, dpi=200)
          clf()

    #-------------------------------------------------------------------------------
    # GHI timeseries
    #-------------------------------------------------------------------------------
    f, ax = subplots(nrows=1, ncols=1, figsize=(9,4))
    drawings.plot_tseries(ax,dates[si],mghi[si,:],"Measurement",
                          b=fghi[si,:], blab="Forecast", d=cghi[si,:],
                          dlab="Clear Sky", slider=interactive, lead=int(180/step),
                          title="GHI 3min horizon", ylabel="Irradiance ($Wm^{-2}$)",
                          sampling="1min")
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_ghi_5.png', dpi=200)
    clf(); close()
    f, ax = subplots(nrows=1, ncols=1, figsize=(9,4))
    drawings.plot_tseries(ax,dates[si],mghi[si,:],"Measurement",
                          b=fghi[si,:], blab="Forecast", d=cghi[si,:],
                          dlab="Clear Sky", slider=interactive, lead=0,
                          title="GHI Nowcast", ylabel="Irradiance ($Wm^{-2}$)",
                          sampling="1min")
    savefig(figpath + os.sep + dtstring + '_' + ststring + '_ghi_0.png', dpi=200)
    clf(); close()
    if interactive:
        ax = subplot(111)
        drawings.plot_tseries_lead(ax, dates[si], mghi[si,:], "Measurement",\
            b=fghi[si,:],blab="Forecast", c=mghi[si,:],clab="Persistence", \
            d=pghi[si,:],dlab="Meas", slider=interactive, step=step,\
            tstamp=int(len(dates[si])/2), sampling="1min", title="GHI", \
            ylabel="Irradiance ($Wm^{-2}$)")
        tight_layout()
        show()
        clf()



    #-------------------------------------------------------------------------------
    # Accuracy
    #-------------------------------------------------------------------------------
    if acc_flag:
        if not quiet: print("Accuracy...")
        # clear sky indizes (global)
        ka = np.divide(mghi[si,:], cghi[si,:])
        kb = np.divide(fghi[si,:], cghi[si,:])
        kp = np.divide(pghi[si,:], cghi[si,:])

        drawings.accuracy_plot(ka, kb, kp, lim=0.7, plttitle="Accuracy GHI",
                               color=color_flag, plot_n=True)
        savefig(figpath + os.sep + dtstring + '_' + ststring + '_accuracy_lead_ghi.'+fmt, fmt=fmt, dpi=200)
        clf()
        accf, accp, ns = drawings.accuracy_time(dates[si], ka, kb, kp, lim=0.7,
                                                plttitle="Accuracy GHI",
                                                color=color_flag)
        savefig(figpath + os.sep + dtstring + '_' + ststring + '_accuracy_time_ghi.'+fmt, fmt=fmt, dpi=200)
        clf()
