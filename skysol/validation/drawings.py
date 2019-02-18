# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
# title	        : drawings.py
# description	: plot functions for skyimager forecast validation
# version		: 0.5
# usage		    :
# notes		    :
# python_version  : 3.5

Created on Mon Dec 28 10:45:18 2015

@author: thomas.schmidt
#==============================================================================

"""
from datetime import datetime, timedelta
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
from matplotlib.widgets import Slider
from skysol.validation.misc import calc_errs  # auxillary functions
from skysol.validation import error_metrics as err # error metric functions
from scipy import stats

import matplotlib.pyplot as plt

# basic matplotlib settings
rcParams['xtick.labelsize'] = 12.
rcParams['ytick.labelsize'] = 12.
rcParams['xtick.color'] = "#000000"
rcParams['ytick.color'] = "#000000"
rcParams['axes.labelcolor'] = "#000000"
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.size'] = 3.0
rcParams['ytick.major.size'] = 3.0
rcParams['figure.facecolor'] = '#FFFFFF'
rcParams['axes.prop_cycle'] = cycler('color',['#348ABD', '#A60628', '#467821', '#7A68A6', '#CF4457', '#188487', '#E24A33', '#556B2F'])
rc('mathtext',default='regular',fontset='cm')

# define default colors

# if gray color tones are used (flag color=False)
g_lcs = ['k','k','k','k','k']
g_msty = ['x','o','y','v']
g_lsty = ['-','--',',-.',':']

# color tones
c_lcs = [ d['color'] for d in list(rcParams['axes.prop_cycle']) ]
c_msty = np.repeat(['o'],len(c_lcs))
c_lsty = np.repeat(['-'],len(c_lcs))

def get_tlabel():

    if rcParams['timezone'] == 'UTC':
        return 'Time (UTC)'
    else:
        return 'Local Time'


def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else gca()
    ax.fill_between(x, y1, y2, **kwargs)
    p = Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)
    return p


def plot_u_sza(ax, u, up, v, color=True):
    """
    Evaluate forecast error vs. pixels viewing angle

    Figure top: Plot MAE vs. Viewing angle (binned instances)
    Figure bottom: Cumulative MAE vs. viewing angle

    """
    bins = np.linspace(0,1.57,70)
    steps = (bins[1] - bins[0])/2.
    digitized = np.digitize(v, bins)
    digitized[np.isnan(v)]=-1
    avg1=[]; avg2=[]; ns=[]; std1=[]
    for i in range(1,len(bins)+1):
        inds = digitized==i#np.where(digitized==i)[0]
        avg1.append(np.nanmean(u[inds]))
        avg2.append(np.nanmean(up[inds]))
        std1.append(np.nanstd(u[inds]))
        ns.append(np.sum(inds))
    ns = np.array(ns)

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty

    ax[0].plot(np.degrees(bins+steps), avg2, lw=2.5, marker="o", c=lcs[1], label="Persistence")
    ax[0].plot(np.degrees(bins+steps), avg1, lw=2.5, marker="o", c=lcs[0], label="Forecast")
    ax[0].set(xlim=(0, 90))
    ax[0].set_ylabel("MAE in $Wm^{-2}$",fontsize=14.,fontweight="bold")
    ax[0].tick_params(axis='x', colors='black')
    ax[0].tick_params(axis='y', colors='black')
    l = ax[0].legend(fancybox=True,loc=2, title='')
    l.get_title().set_fontsize('13')
    l.get_title().set_fontweight('bold')

    ax2 = ax[0].twinx()
    ax2.plot(np.degrees(bins+steps),ns/1000,"-.",c='black',lw=1.5)
    ax2.set(xlim=(0, 90))
    ax2.locator_params(nbins=6)
    ax2.grid('off')
    ax2.set_ylabel("Number of instances in thousands")

    ax[0].grid('on')

    allns = np.sum(ns)
    cum1 = np.cumsum(np.float32(avg1)*ns/allns)
    cum2 = np.cumsum(np.float32(avg2)*ns/allns)

    ax[1].plot(np.degrees(bins+steps),cum2, "-", c=lcs[1], lw=2.5, label="Persistence")
    ax[1].plot(np.degrees(bins+steps),cum1, "-", c=lcs[0], lw=2.5, label="Forecast")
    ax[1].set_ylabel("Cumulative weighted MAE in $Wm^{-2}$", fontsize=14., fontweight="bold")
    ax[1].set_xlabel("Viewing angle in deg", fontsize=14., fontweight="bold")
    l = ax[1].legend(fancybox=True,loc=2, title='')
    l.get_title().set_fontsize('13')
    l.get_title().set_fontweight('bold')
    ax[1].set(xlim=(0, 90))
    tight_layout()

    del digitized


def plot_uv_scatter(ax,u,up,v,clclass=[], legend=True):
    """
    Plot scattered uncertainty vs. variability
    """
    from scipy import stats

    if len(clclass) == 0:
        c = "#990000"
    else:
        c = "black"
    ax.scatter(v,up,marker="o",c=c,s=12.,alpha=0.8,label="Persistence")
    ax.plot([0,0.9],[0,0.9],ls="--",c="k",lw=1.0,label="1:1")
    ax.set(xlim=(0, 0.9), ylim=(0, 0.9))
    if len(clclass) == 0:
        c = "#0066FF"
    else:
        c = ["#9b59b6", "#2c6fbb", "#b04e0f", "#028f1e", '#A90016', "#34495e", "#ffad01"]
    sc = ax.scatter(v,u,marker="o",alpha=0.9,s=12.,c=c,label="Forecast")
    if type(c) == np.ndarray:
        colorbar(sc,cax=ax)
    ax.set_xlabel("V",fontsize=15.,fontweight="bold")
    ax.set_ylabel("U",fontsize=15.,fontweight="bold")
    ax.grid('off')

    inds = np.isfinite(u) & np.isfinite(v)
    slope, intercept, r_value, p_value, std_err = stats.linregress(v[inds],u[inds])
    x = np.arange(0,1,1/100.)
    S = 1- np.nanmean(u/v)
    c = "#000000"
    ax.plot(x,x*slope + intercept,c=c,ls="-.",lw=1.5,label="Forecast Fit")
    ax.text(0.99,0.05,('f(V) = %.2f * V + %.2f\n$R^2$ = %.2f' % (slope, intercept,r_value**2)),\
        fontsize=10.,horizontalalignment="right",transform=ax.transAxes)
    if legend: ax.legend(fancybox=True,loc=2,fontsize=7.)
    ax.tick_params(axis='both', colors='black', labelsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)



def plot_tseries(ax, dates, a, alab, b=None, blab="", c=None, clab="", \
    d=None, dlab="", slider=False, title="", lead=0, sampling="", \
    ylabel="", color=True):
    """
    Plot daily timeseries of measurements and forecasts for a given lead time.
    The lead can be changed interactively with the slider keyword.

    The temporal resolution can be changed with the sampling keyword. Time series are
    averaged to the given resolution.
    """
    from pandas import Series
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty

    if sampling == "":
        sampling = '%dS' % (dates[1].second - dates[0].second)

    def draw_stat(ax,e,yshift=0,c="black"):

        y = 0.2 * yshift
        textkeys = dict(fontsize=9, horizontalalignment="left",
            backgroundcolor="white",
            verticalalignment="top",fontweight="bold", color=c)
        ax.text(0.02,0.90-y,('RMSE = %.1f' % \
            (np.nanmean(e['rmse']))),transform=ax.transAxes, **textkeys)
        ax.text(0.02,0.97-y,('MBE = %.1f' % \
            (np.nanmean(e['mbe']))),transform=ax.transAxes, **textkeys)
        ax.text(0.02,0.84-y,('R = %.2f' % \
            (np.nanmean(e['corr']))),transform=ax.transAxes, **textkeys)

    Locator = AutoDateLocator()
    dateform = AutoDateFormatter(Locator)
    dateform.scaled[1/(24.)] = '%H'
    dateform.scaled[1. /(24. * 60.)] = '%H:%M'

    a = np.float32(a)
    a1 = Series(a[:,lead], index=dates).resample(sampling, label="right").mean()

    dts = a1.index.to_pydatetime()
    at = plot(dts, a1, label=alab, color='k', linestyle=lsty[0])


    if b is not None:
        b = np.float32(b)
        b1 = Series(b[:,lead], index=dates).resample(sampling, label="right").mean()
        bt = plot(dts,b1,label=blab, color=lcs[0], linestyle=lsty[0],lw=1)
    if c is not None:
        c = np.float32(c)
        c1 = Series(c[:,lead], index=dates).resample(sampling, label="right").mean()
        ct = plot(dts,c1,label=clab, color=lcs[1], linestyle=lsty[1],lw=1)
    if d is not None:
        d = np.float32(d)
        d1 = Series(d[:,lead], index=dates).resample(sampling, label="right").mean()
        dt = plot(dts,d1,label=dlab, color=lcs[2], linestyle=lsty[2],lw=1)
    if title != "":
        ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(dateform)
    ax.xaxis.set_major_locator(Locator)
    ax.set_ylabel(ylabel, fontsize="large")
    ax.set_xlabel(get_tlabel(),fontsize="large")

    e = calc_errs(a[:,lead],b[:,lead],taxis=0)
    draw_stat(ax,e,c=lcs[0])
    if c is not None:
        e = calc_errs(a[:,lead],c[:,lead],taxis=0)
        draw_stat(ax,e,yshift=1,c=lcs[1])

    legend(loc='upper right', fontsize="small")
    if slider:
        ax_s = axes([0.25, 0.01, 0.65, 0.03])
        slid = Slider(ax_s,   'Lead',   0, a.shape[1]-1, valinit=lead, valfmt="%d")
        def update(val):
            lead    =    int(slid.val)
            at[0].set_ydata( Series(a[:,lead], index=dates).resample(sampling, label="right").mean() )
            if b is not None: bt[0].set_ydata( Series(b[:,lead], index=dates).resample(sampling, label="right").mean())
            if c is not None: ct[0].set_ydata( Series(c[:,lead], index=dates).resample(sampling, label="right").mean())
            if d is not None: dt[0].set_ydata( Series(d[:,lead], index=dates).resample(sampling, label="right").mean())
            e = calc_errs(a[:,lead],b[:,lead], taxis=0)
            draw_stat(ax,e,c=lcs[0])
            if c is not None:
                e = calc_errs(a[:,lead],c[:,lead],taxis=0)
                draw_stat(ax,e,yshift=1,c=lcs[2])
            draw()
        slid.on_changed(update)





def plot_tseries_lead(ax,dates, a, alab, b=None, blab="", c=None, clab="", \
    d=None, dlab="", slider = False, title="", tstamp=0, step=1, sampling="10s", \
    ylabel="", color=True, csk=None):
    """ Plot daily time series of measurements. Add forecasts for single timesteps.
    Using the slider option, the forecasts for the whole day can be visualized easily
    """

    from pandas import Series
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty


    def draw_stat(ax,e,yshift=0,c="black"):

        y = 0.2 * yshift
        textkeys = dict(fontsize=14,horizontalalignment="left",
            backgroundcolor = "white",
            verticalalignment="top",fontweight="bold", color=c)
        ax.text(0.02,0.92-y,('RMSE = %.1f' % \
            (np.nanmean(e['rmse']))),transform=ax.transAxes, **textkeys)
        ax.text(0.02,0.98-y,('MBE = %.1f' % \
            (np.nanmean(e['mbe']))),transform=ax.transAxes, **textkeys)
        ax.text(0.02,0.86-y,('R = %.2f' % \
            (np.nanmean(e['corr']))),transform=ax.transAxes, **textkeys)


    Locator = AutoDateLocator()
    dateform = AutoDateFormatter(Locator)
    dateform.scaled[1/(24.)] = '%H'
    dateform.scaled[1. /(24. * 60.)] = '%H:%M'

    hor = a.shape[1]
    ds = dates[tstamp]
    dts = [ ds + timedelta(seconds=i*step) for i in range(0,hor) ]

    if sampling != "":
        a1 = Series(a[:,0], index=dates).resample(sampling, label="right").mean()
        dt = a1.index.to_pydatetime()
        a1.plot(label=alab, color='k', linestyle=lsty[0])
    else:
        plot(dates, a[:,0], label=alab, color='k', linestyle=lsty[0])

    if csk is not None:
        plot(dates, csk[:,0], label="Clear Sky", color="k", linestyle="--")
    if b is not None: bt = plot(dts,b[tstamp,:], label=blab, color=lcs[0], linestyle=lsty[0])
    if c is not None: ct = plot(dts,c[tstamp,:], label=clab, color=lcs[1], linestyle=lsty[1])
    if d is not None: dt = plot(dts,d[tstamp,:], label=dlab, color=lcs[2], linestyle=lsty[2])

    if title != "":
        ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(dateform)
    ax.xaxis.set_major_locator(Locator)
    ax.set_ylabel(ylabel, fontsize="x-large")
    ax.set_xlabel(get_tlabel(),fontsize="x-large")

    e = calc_errs(a[tstamp,:],b[tstamp,:],taxis=0)
    draw_stat(ax,e,c=lcs[0])
    if c is not None:
        e = calc_errs(a[tstamp,:],c[tstamp,:],taxis=0)
        draw_stat(ax,e,yshift=1,c=lcs[1])

    legend(loc='upper right')

    if slider:
        ax_s = axes([0.25, 0.01, 0.65, 0.03])
        slid = Slider(ax_s,   'Timestamp',   0, len(dates), valinit=tstamp, valfmt="%d")

        def update(val):
            tstamp    =    int(slid.val) # start

            ds = dates[tstamp]
            dts = [ ds + timedelta(seconds=i*step) for i in range(0,hor) ]

            #at[0].set_ydata( Series(a[tstamp,:], index=dates).resample(sampling, label="right").mean() )
            if b is not None:
                bt[0].set_xdata(dts)
                bt[0].set_ydata(b[tstamp,:])
            if c is not None:
                ct[0].set_xdata(dts)
                ct[0].set_ydata(c[tstamp,:])
            if d is not None:
                dt[0].set_xdata(dts)
                dt[0].set_ydata(d[tstamp,:])
            e = calc_errs(a[tstamp,:],b[tstamp,:],taxis=0)
            draw_stat(ax,e,c=lcs[0])
            if c is not None:
                e = calc_errs(a[tstamp,:],c[tstamp,:],taxis=0)
                draw_stat(ax,e,yshift=1,c=lcs[2])


            draw()

        slid.on_changed(update)



def plot_tseries_eval(dates, mghi, fghi, cghi, pghi,lead=0, titlestring="", \
    slider=False, sampling="", averaging="", color=True):
    """
    Evaluates forecasted timeseries against measurements and compares with a
    reference forecast ( e.g. persistence ).

    It calculates error metrics for each forecast ( along the forecast horizon ).
    Consequently the result is a timeseries of the error metrics.

    It calculates correlation, rmse, bias and skill and visualizes in a graphics.

    The window for which the error metrics should be calculated is given by
    the optional argument "lead". If it is not given, the whole forecast horizon
    is used for the error calculation.

    Optionally, timeseries can be resampled ( down-sampling or up-sampling )
    before the error calculation. It is also possible to average the final
    error_metrics for different temporal resolutions.

    A slider can be used to interactively change evaluated lead time or
    time series window.
    """


    from matplotlib.dates import AutoDateFormatter, AutoDateLocator, num2date, date2num
    from pandas import Series

    horizon = mghi.shape[1]

    dts = dates

    if lead == 0: lead = slice(0,horizon,None)


    # Resampling ( down- or upsample input timeseries before error calculation )
    if sampling != "":
        # Resample time stamps
        dts = Series(mghi[:,0], index=dates).resample(sampling, label="right").mean().index.to_pydatetime()

        # Resample data
        a = np.empty((len(dts),horizon))
        b = np.empty((len(dts),horizon))
        c = np.empty((len(dts),horizon))
        d = np.empty((len(dts),horizon))

        for i in range(0,horizon):
            a[:,i] = Series(mghi[:,i], index=dates).resample(sampling, label="right").mean()
            b[:,i] = Series(fghi[:,i], index=dates).resample(sampling, label="right").mean()
            c[:,i] = Series(cghi[:,i], index=dates).resample(sampling, label="right").mean()
            d[:,i] = Series(pghi[:,i], index=dates).resample(sampling, label="right").mean()

        mghi = a; fghi = b; cghi = c; pghi = d


    # remove persistence values where no forecast available
    pghi[~np.isfinite(fghi)] = np.nan

    # calculate error metrics
    e = calc_errs(mghi[:,lead],fghi[:,lead],pghi[:,lead],taxis=1)
    ep = calc_errs(mghi[:,lead],pghi[:,lead],taxis=1)


    # Averaging the final error metrics
    if averaging != "":

         # Resample time stamps
        dts = Series(e['rmse'], index=dates).resample(averaging, label="right").mean().index.to_pydatetime()
        # Resample data
        a = np.empty((len(dts),horizon))

        for key in ['rmse','mae','fs','mbe','corr']:
            e[key] = Series(e[key], index=dates).resample(averaging, label="right").mean()
            try:
                ep[key] = Series(ep[key], index=dates).resample(averaging, label="right").mean()
            except KeyError:
                pass
        for i in range(0,e['nvalid'].shape[1]):
            a[:,i] = Series(e['nvalid'][:,i], index=dates).resample(averaging, label="right").mean()
        e['nvalid'] = a

    labelsize = 8

    # average observation value for relative errors
    meanobs = np.nanmean(mghi[:,lead],dtype=np.float32)

    # matplotlib settings
    textkeys = dict(fontsize=7, horizontalalignment="left",
                verticalalignment="top", fontweight="bold")

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty

    # Date / Xtick Labeling
    Locator = AutoDateLocator(minticks=3, maxticks=10)
    dateform = AutoDateFormatter(Locator)
    dateform.scaled[1/(24.)] = '%H'
    dateform.scaled[1. /(24. * 60.)] = '%H:%M'
    close('all')
    fig = Figure(figsize=(11,12))

    def draw_graphs(x,e,ep,titlestring=""):

        if titlestring == "":
            if x[0].day != x[-1].day:
                start = datetime.strftime(x[0],"%Y-%m-%d")
                end = datetime.strftime(x[-1],"%Y-%m-%d")
                t = start + ' to ' + end
            else:
                t = datetime.strftime(x[0],"%Y-%m-%d")
            titlestring = "Forecast Performance for " + t
        fig.suptitle(titlestring, horizontalalignment="center", fontsize=20)

        # Correlation
        ax1 = subplot(511)
        ax1.set_title('Correlation', fontweight="bold", fontsize="small")
        ax1.plot(x,np.repeat(0,len(x)),'k--',lw=1.6)
        ax1.plot(x,ep['corr'], label="Persistence", lw=1.3, ls=lsty[1], c=lcs[1])
        ax1.plot(x,e['corr'], label="Forecast", lw=1.6, ls=lsty[0], c=lcs[0])
        ax1.xaxis.set_major_locator(Locator)
        ax1.xaxis.set_major_formatter(dateform)
        ax1.set_ylim(-1,1)

        ax1.text(0.01, 1.2,('avg. %.2f' % np.nanmean(e['corr'])), transform=ax1.transAxes, color=lcs[0], **textkeys)
        ax1.text(0.1, 1.2,('%.2f' % np.nanmean(ep['corr'])), transform=ax1.transAxes, color=lcs[1], **textkeys)
        ax1.tick_params(axis="both", labelsize=labelsize, labelbottom='off')
        ax1.legend(fontsize=7)

        # RMSE
        ax2 = subplot(512,sharex=ax1)
        ax2.set_title('RMSE', fontweight="bold", fontsize="small")
        ax2.plot(x,ep['rmse'],label="Persistence",lw=1.5, ls=lsty[1], c=lcs[1])
        ax2.plot(x,e['rmse'],label="Forecast",lw=1.6, ls=lsty[0], c=lcs[0])
        ax2.xaxis.set_major_locator(Locator)
        ax2.xaxis.set_major_formatter(dateform)
        ax2.set_ylabel("$Wm^{-2}$", fontsize=labelsize)
        ax2a = ax2.twinx()
        ax1Ys = ax2.get_yticks()
        ax2Ys = []
        for Y in ax1Ys:
            ax2Ys.append(str(round(100. * Y / meanobs,0)))
        ax2a.set_yticks(ax1Ys); ax2a.set_ybound(ax2.get_ybound()); ax2a.set_yticklabels(ax2Ys)
        ax2a.set_ylabel("%", fontsize=labelsize)
        ax2a.tick_params(axis="both", labelsize=labelsize)
        ax2.text(0.01, 1.2,('avg. %.2f (%.1f %%)' % \
            (np.nanmean(e['rmse']) , 100. * np.nanmean(e['rmse']) / meanobs)), \
            transform=ax2.transAxes, color=lcs[0],  **textkeys)
        ax2.text(0.2, 1.2,('%.2f (%.1f %%)' % \
            (np.nanmean(ep['rmse']), 100. * np.nanmean(ep['rmse']) / meanobs)), \
            transform=ax2.transAxes, color=lcs[1], **textkeys)

        ax2.tick_params(axis="both", labelsize=labelsize, labelbottom='off')

        # Bias
        ax3 = subplot(513,sharex=ax1)
        ax3.set_title('Bias', fontweight="bold", fontsize="small")
        ax3.plot(x,np.repeat(0,len(x)),'k--',lw=1.6)
        ax3.plot(x,ep['mbe'],label="Persistence",lw=1.5, ls=lsty[1], c=lcs[1])
        ax3.plot(x,e['mbe'],label="Forecast",lw=1.6, ls=lsty[0], c=lcs[0])
        ax3.xaxis.set_major_locator(Locator)
        ax3.xaxis.set_major_formatter(dateform)

        ax3.set_ylabel("$Wm^{-2}$", fontsize=labelsize)

        ax3.text(0.01,1.2, ('avg. %.2f (%.1f %%)'  % \
            (np.nanmean(e['mbe']) , 100. * np.nanmean(e['mbe']) / meanobs )),\
                transform=ax3.transAxes, color=lcs[0], **textkeys)
        ax3.text(0.2,1.2, ('%.2f (%.1f %%)' % \
            (np.nanmean(ep['mbe']), 100. * np.nanmean(ep['mbe']) / meanobs)),\
                transform=ax3.transAxes, color=lcs[1], **textkeys)

        ax3.tick_params(axis="both", labelsize=labelsize, labelbottom='off')

        # Forecast Skill
        ax4 = subplot(514,sharex=ax1)
        ax4.set_title('Forecast Skill', fontweight="bold", fontsize="small")
        ax4.plot(x,np.repeat(0,len(x)),'k--',lw=1.6, ls="--", c="k")
        ax4.plot(x,e['fs'], lw=1.6, c=lcs[1], label="model")
        ax4.xaxis.set_major_formatter(dateform)
        ax4.xaxis.set_major_locator(Locator)
        ax4.set_ylim(-1,1)
        ax4.text(0.01,1.2,('avg. Skill= %.2f' % (np.nanmean(e['fs']))), \
            transform=ax4.transAxes, color=lcs[1], **textkeys)
        ax4.tick_params(axis="both", labelsize=labelsize, labelbottom='off')

        # Number of instances
        ax5 = subplot(5,1,5,sharex=ax1)
        ax5.set_title('Number of instances', fontweight="bold", fontsize="small")
        ax5.plot(x,(e['nvalid'] != 0).sum(axis=1),lw=1.6, ls=lsty[1], c=lcs[1])
        ax5.xaxis.set_major_formatter(dateform)
        ax5.xaxis.set_major_locator(Locator)
        ax5.tick_params(axis="both", labelsize=labelsize)
        ax5.set_xlabel(get_tlabel())

        draw()

        return (ax1, ax2, ax2a, ax3, ax4, ax5)


    all_axes = draw_graphs(dts, e, ep, titlestring = titlestring)


    if slider:


        sld1 = axes([0.15, 0.055, 0.8, 0.015])
        sld2 = axes([0.15, 0.04, 0.8, 0.015])
        sld3 = axes([0.15, 0.025, 0.8, 0.015])
        sld4 = axes([0.15, 0.01, 0.8, 0.015])

        sdate = date2num(dates[0])
        edate = date2num(dates[-11])

        slid1 = Slider(sld1,   'End Lead',   lead.stop, fghi.shape[1]-1, valinit=0, valfmt="%d")
        slid2 = Slider(sld2,   'Start Lead',   lead.start, fghi.shape[1]-1, valinit=fghi.shape[1]-1, valfmt="%d")

        sdatesld = Slider(sld3,   'Start Time',   sdate, edate, valinit = sdate)
        edatesld = Slider(sld4,   'End Time',   sdate, edate, valinit = edate)

        # update slices
        def update(val):
            lead0   =    int(slid1.val)
            lead1   =    int(slid2.val)
            lead = slice(lead0,lead1,None)
            x0 = num2date(sdatesld.val).replace(tzinfo=None)
            x1 = num2date(edatesld.val).replace(tzinfo=None)
            inds = (np.array(dates) > x0) & (np.array(dates) < x1)
            x0 = np.where(inds)[0][0]
            x1 = np.where(inds)[0][-1]
            slc = slice(x0,x1, None)
            e = calc_errs(mghi[slc,lead],fghi[slc,lead],pghi[slc,lead],taxis=1)
            ep = calc_errs(mghi[slc,lead],pghi[slc,lead],taxis=1)

            dts = dates[slc]


            if averaging != "":


                 # Resample time stamps
                dts = Series(e['rmse'], index=dates[slc]).resample(averaging, label="right").mean().index.to_pydatetime()
                # Resample data
                a = np.empty((len(dts),horizon))

                for key in ['rmse','mae','fs','mbe','corr']:
                    e[key] = Series(e[key], index=dates[slc]).resample(averaging, label="right").mean()
                    try:
                        ep[key] = Series(ep[key], index=dates[slc]).resample(averaging, label="right").mean()
                    except KeyError:
                        pass
                for i in range(0,e['nvalid'].shape[1]):
                    a[:,i] = Series(e['nvalid'][:,i], index=dates[slc]).resample(averaging, label="right").mean()
                e['nvalid'] = a


            # Clean old figure
            for ax in all_axes:
                ax.cla()

            # Draw new figure
            ax = draw_graphs(dts, e, ep , titlestring = titlestring)

            return ax


        ax = slid1.on_changed(update)
        ax = slid2.on_changed(update)
        ax = sdatesld.on_changed(update)
        ax = edatesld.on_changed(update)



def plot_lead_eval(dates, mghi, fghi, cghi, pghi, step=1, titlestring="", color=True):


    horizon = mghi.shape[1]
    # remove persistence values where no forecast available
    pghi[~np.isfinite(fghi)] = np.nan
    x = np.arange(1,mghi.shape[1]+1)
    # calculate error metrics
    e = calc_errs(mghi,fghi,pghi,taxis=0)
    ep = calc_errs(mghi,pghi,taxis=0)

    # average observation value for relative errors
    meanobs = np.nanmean(mghi[:,0],dtype=np.float32)

    labelsize=7

    textkeys = dict(fontsize=7,horizontalalignment="left",
                        verticalalignment="top",fontweight="bold")

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty

    fig, axes = subplots(nrows=5,ncols=1,figsize=(7,6))
    # Correlation
    ax = axes[0]
    fig.text(0.5,0.95, titlestring, horizontalalignment="center", fontsize=20)
    title('Correlation', fontweight="bold", fontsize="small")
    ax.plot(x,np.repeat(0,horizon),'k--',lw=1.7)
    ax.plot(x,ep['corr'],label="Persistence",lw=1.2,ls=lsty[1],c=lcs[1])
    ax.plot(x,e['corr'],label="Forecast",lw=1.7,ls=lsty[0],c=lcs[0])
    ax.set_ylim(-1,1)
    ax.tick_params(axis='both', which='major', labelsize=8, labelbottom='off')
    ax.text(0.01, 1.15,('avg. %.2f' % np.nanmean(e['corr'])), transform=ax.transAxes, color=lcs[0], **textkeys)
    ax.text(0.15, 1.15,('%.2f' % np.nanmean(ep['corr'])), transform=ax.transAxes, color=lcs[1], **textkeys)
    xlim(1,horizon)
    ax.legend(loc="lower right", fontsize=6)

    # RMSE
    ax = axes[1]
    ax.set_title('RMSE', fontweight="bold", fontsize="small")
    ax.plot(x,ep['rmse'],label="Persistence",lw=1.2,ls=lsty[1],c=lcs[1])
    ax.plot(x,e['rmse'],label="Forecast",lw=1.7,ls=lsty[0],c=lcs[0])
    ax.tick_params(axis='both', which='major', labelsize=8, labelbottom='off')
    ax.set_ylabel("$Wm^{-2}$", fontsize=labelsize)

    ax2 = ax.twinx()
    ax1Ys = ax.get_yticks()
    ax2Ys = []
    for Y in ax1Ys:
        ax2Ys.append(str(round(100. * Y / meanobs,0)))
    ax2.set_yticks(ax1Ys); ax2.set_ybound(ax.get_ybound()); ax2.set_yticklabels(ax2Ys)
    ax2.set_ylabel("%", fontsize=labelsize)
    ax2.tick_params(axis='both', which='major', labelsize=9)

    ax.text(0.01, 1.15,('avg. %.2f (%.1f %%)' % \
        (np.nanmean(e['rmse']) , 100. * np.nanmean(e['rmse']) / meanobs)), \
        transform=ax.transAxes, color=lcs[0],  **textkeys)
    ax.text(0.25, 1.15,('%.2f (%.1f %%)' % \
        (np.nanmean(ep['rmse']), 100. * np.nanmean(ep['rmse']) / meanobs)), \
        transform=ax.transAxes, color=lcs[1], **textkeys)
    ax.set_xlim(1,horizon)
    ax.tick_params(axis='both', which='major', labelsize=8, labelbottom='off')




    # Bias
    ax = axes[2]
    ax.set_title('Bias', fontweight="bold", fontsize="small")
    ax.plot(x,np.repeat(0,horizon),'k--', lw=1.7)
    ax.plot(x,ep['mbe'],label="Persistence", lw=1.2,ls=lsty[1],c=lcs[1])
    ax.plot(x,e['mbe'],label="Forecast", lw=1.7,ls=lsty[0],c=lcs[0])
    ax.set_ylabel("$Wm^{-2}$", fontsize=labelsize)
    ax.tick_params(axis='both', which='major', labelsize=8, labelbottom='off')

    ax2 = ax.twinx()
    ax1Ys = ax.get_yticks()
    ax2Ys = []
    for Y in ax1Ys:
        ax2Ys.append(str(round(100. * Y / meanobs,0)))
    ax2.set_yticks(ax1Ys); ax2.set_ybound(ax.get_ybound()); ax2.set_yticklabels(ax2Ys)
    ax2.set_ylabel("%", fontsize=labelsize)
    ax2.tick_params(axis='both', which='major', labelsize=8.5)

    ax.text(0.01,1.15, ('avg. %.2f (%.1f %%)'  % \
        (np.nanmean(e['mbe']) , 100. * np.nanmean(e['mbe']) / meanobs )),\
            transform=ax.transAxes, color=lcs[0], **textkeys)
    ax.text(0.25,1.15, ('%.2f (%.1f %%)' % \
        (np.nanmean(ep['mbe']), 100. * np.nanmean(ep['mbe']) / meanobs)),\
            transform=ax.transAxes, color=lcs[1], **textkeys)

    ax.set_xlim(1,horizon)


    ax = axes[3]
    ax.set_title('Forecast Skill', fontweight="bold", fontsize="small")
    ax.plot(x,np.repeat(0,horizon),'k--',lw=1.5)
    ax.plot(x,e['fs'],lw=1.5,label="model",c=lcs[1])
    ax.set_ylim(-1,1)
    ax.tick_params(axis='both', which='major', labelsize=8, labelbottom='off')
    la = int(61 / step)
    avg_fs = np.nanmean(e['fs'][la:])
    ax.text(0.01,1.15,('avg. Skill (1min+) = %.2f' % (avg_fs)),transform=ax.transAxes, \
        color = lcs[1], **textkeys)
    ax.set_xlim(1,horizon)


    ax = axes[4]
    ax.set_title('Number of instances', fontweight="bold", fontsize="small")
    ax.tick_params(axis='both', which='major', labelsize=8.5)
    ax.plot(x,(e['nvalid'] != 0).sum(axis=0),lw=1.5,c=lcs[1])
    ax.set_xlim(1,horizon)
    ax.set_xlabel('Forecast horizon')

    fig.tight_layout()





def draw_scatter(ax, x, y, y2=None, plttitle="", lead=0, param_name="", \
        ylab="", y2lab="", interactive=True, color=True):
    """
    Draw simple scatter plot for Measurement vs. Model.
    Two model variables can be specified.
    Can be used interactively with variable lead time
    """

    if interactive and x.ndim==1:
        print('draw_scatter: Provide two-dimensional arrays for interactive scatter plots!')
        return

    x = np.float32(x)
    y = np.float32(y)
    if y2 is not None:
        y2 = np.float32(y2)
        if y2.ndim == 1:
            y2t = y2[:]
        else:
            y2t = y2[:,lead]
    else:
        y2t = None

    if color == False:
        lcs = ['k','k','k','k','k']
        lsty = ['x','o','y','v']
    else:
        lcs = [ d['color'] for d in list(rcParams['axes.prop_cycle']) ]
        lsty = np.repeat(['o'],len(lcs))

    def draw(x,y,y2=None):

        # average observations
        meanobs = np.nanmean(x)

        # plot first variable
        ax.scatter(x, y, s=3, color=lcs[0], marker=lsty[0])
        # plot second
        if y2 is not None: ax.scatter(x, y2, s=2, color=lcs[1], marker=lsty[1])
        xmax = np.nanmax(x); ymax = np.nanmax(y)
        amax = np.nanmax([xmax,ymax])
        amax += 0.1*amax
        # plot diagonal
        ax.plot(np.arange(0,amax),np.arange(0,amax), 'k--', lw=2.0)
        ax.set_title(plttitle, fontweight="bold", fontsize='large')
        ax.set_xlabel('Measurement '+ param_name, fontsize='medium', fontweight="bold")
        ax.set_ylabel('Model ' + param_name, fontsize='medium', fontweight="bold")
        ax.set_xlim(0,amax)
        ax.set_ylim(0,amax)
        ax.grid('on')
        e = calc_errs(x,y)
        if type(e) == np.float: return
        r = e['corr']
        rmse = e['rmse']
        bias = e['mbe']
        ax.grid("on")
        # Error statistics
        ax.text(0.02,0.98,(ylab+'\n# Data points = %d\nRMSE = %.2f (%.1f %%)\nBias = %.2f (%.1f %%)\nCorr = %.3f' % (np.sum(np.isfinite(y)), rmse, 100*rmse/meanobs, bias, 100*bias/meanobs, r)),\
            fontsize='x-small', backgroundcolor="white",  verticalalignment="top",horizontalalignment="left",transform=ax.transAxes, color=lcs[0])
        if y2 is not None:
            e = calc_errs(x,y2)
            r = e['corr']
            rmse = e['rmse']
            bias = e['mbe']
            ax.text(0.02,0.80,(y2lab+'\n# Data points = %d\nRMSE = %.2f (%.1f %%)\nBias = %.2f (%.1f %%)\nCorr = %.3f' % (np.sum(np.isfinite(y2)), rmse, 100*rmse/meanobs, bias, 100*bias/meanobs, r)),\
            fontsize='x-small', backgroundcolor="white", color=lcs[1], verticalalignment="top",horizontalalignment="left",transform=ax.transAxes)

    if x.ndim == 1:
        draw(x[:], y[:], y2=y2t)
    else:
        draw(x[:,lead], y[:,lead], y2=y2t)

    # Interactive mode
    if interactive:
        slidax = axes([0.25, 0.01, 0.65, 0.03])
        slid = Slider(slidax,   'Lead',   0, x.shape[1], valinit=lead, valfmt="%d")

        def update(val):
            tmp    =    int(slid.val)
            xt = x[:,tmp]
            yt = y[:,tmp]
            if y2 is not None: y2t  = y2[:,tmp]
            # Clean old figure
            ax.cla()
            draw(xt,yt,y2=y2t)

        slid.on_changed(update)










def accuracy_plot(obs, fcst, ref, lim=0.7, plttitle="", color=True, axlim=None, \
    plot_n=False, min_val=0.2):
    """
    Plots forecast accuracy (binary statistics) vs forecast lead time.

    It uses clear sky indizes of measurement, forecast and a reference forecast
    (persistence). It computes the boolean value (clear sky index > threshold) and
    computes the accuracy (matched forecasts / all forecasts)
    """
    accf = []; accp = []; ns = []

    from skysol.validation.misc import accuracy

    if color == False:
        lcs = ['k','k','k','k','k']
        lsty = ['-','--',',-.',':']
    else:
        lcs = [ d['color'] for d in list(rcParams['axes.prop_cycle']) ]
        lsty = np.repeat(['-'],len(lcs))

    horizon = obs.shape[1]

    if horizon > 100:
        step = 10
    else:
        step = 1

    # binary forecast
    for i in range(0,horizon,step):
        ind = np.isfinite(obs[:,i]) & np.isfinite(fcst)[:,i] & np.isfinite(ref[:,i])
        if np.sum(ind)/len(ind) > min_val:
            ns.append(np.sum(ind)) # number of instances
            xbin = obs >= lim
            ybin = fcst >= lim
            accf.append(accuracy(ybin[ind,i],xbin[ind,i],taxis=0))
            ybin = ref >= lim
            accp.append(accuracy(ybin[ind,i],xbin[ind,i],taxis=0))
        else:
            ns.append(np.sum(ind))
            accf.append(np.nan)
            accp.append(np.nan)

    f, ax = subplots(figsize=(8,4))
    ax.set_title(plttitle, fontsize="x-large", fontweight="bold")
    x = np.arange(1, horizon+1, step)
    l2 = ax.plot(x,accp,'-x', label="Persistence",mew=2.5,linewidth=2.5, linestyle=lsty[1], color=lcs[1])
    l1 = ax.plot(x,accf, '-x',label="Forecast",mew=2.5,linewidth=2.5, linestyle=lsty[0], color=lcs[0])

    ax.set_xlabel("Forecast horizon (min)", fontsize="large", fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize="large", fontweight="bold")
    if axlim:
        ax.set_ylim(axlim,1)
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0],1)
    ax.set_xlim(1,horizon)
    lns = l1+l2

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',         # ticks along the top edge are off
        right="off")

    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right="off")

    ax.set_ylim(0.8,1)

    if plot_n:
        ax2 = ax.twinx()
        ax2.grid('off')
        l3 = ax2.plot(x,ns,'k:', label="Number of forecasts", linewidth=1.5)
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax2.set_ylabel('Number of forecasts')
        lns = lns+l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, fontsize="medium")
    if plot_n:
        subplots_adjust(left=0.1,right=0.9, bottom=0.15)
    else:
        subplots_adjust(left=0.13,right=0.98, bottom=0.2)




def accuracy_time(dates, obs, fcst, ref, lim=0.7, plttitle="", color=True,
                  axlim=None, plot_n=False, min_val=0.2):
    accf = []; accp = []; ns = []; dts = []

    from skysol.validation.misc import accuracy
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    Locator = AutoDateLocator()
    dateform = AutoDateFormatter(Locator)
    dateform.scaled[1/(24.)] = '%H'
    dateform.scaled[1. /(24. * 60.)] = '%H:%M'

    if color == False:

        lcs = ['k','k','k','k','k']
        lsty = ['-','--',',-.',':']

    else:

        lcs = [ d['color'] for d in list(rcParams['axes.prop_cycle']) ]
        lsty = np.repeat(['-'],len(lcs))



    horizon = obs.shape[0]

    if horizon > 100:
        step = 10
    else:
        step = 1
    # binary forecast
    for i in range(0,horizon,step):
        ind = np.isfinite(obs[i,:]) & np.isfinite(fcst)[i,:] & np.isfinite(ref[i,:])
        if np.sum(ind)/len(ind) > min_val:
            ns.append(np.sum(ind)) # number of instances
            xbin = obs >= lim
            ybin = fcst >= lim
            accf.append(accuracy(ybin[i,ind],xbin[i,ind],taxis=0))
            ybin = ref >= lim
            accp.append(accuracy(ybin[i,ind],xbin[i,ind],taxis=0))
            dts.append(dates[i])
        else:
            ns.append(np.sum(ind))
            accf.append(np.nan)
            accp.append(np.nan)
            dts.append(dates[i])

    f, ax = subplots(figsize=(8,4))
    ax.set_title(plttitle, fontsize="x-large", fontweight="bold")
    l2 = ax.plot(dts,accp,label="Persistence",linewidth=1.5, linestyle=lsty[1], color='gray')
    l1 = ax.plot(dts,accf,label="Forecast", linewidth=2, linestyle=lsty[0], color=lcs[0])
    ax.xaxis.set_major_locator(Locator)
    ax.xaxis.set_major_formatter(dateform)
    ax.set_xlabel(get_tlabel(), fontsize="large", fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize="large", fontweight="bold")
    if axlim:
        ax.set_ylim(axlim,1)
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0],1)
    ax.set_xlim(dts[0],dts[-1])
    lns = l1+l2


    if plot_n:
        ax2 = ax.twinx()
        ax2.grid('off')
        l3 = ax2.plot(dts,ns,'k:', label="Number of forecasts", linewidth=1)
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax2.set_ylabel('Number of forecasts')
        lns = lns+l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs,  fontsize="medium", loc="lower left")
    if plot_n:
        subplots_adjust(left=0.1,right=0.9, bottom=0.15)
    else:
        subplots_adjust(left=0.1,right=0.98, bottom=0.15)


    return accf, accp, ns


def accuracy_thres(obs, fcst, ref, var, slim=np.linspace(0,0.4,10),lim=0.7, plot_n=True, plttitle="", color=True):
    from skysol.validation.misc import accuracy, interpol

    if color:
        lcs = c_lcs; lsty = c_lsty
    else:
        lcs = g_lcs; lsty = g_lsty
    msty = ['x','o','','']


    horizon = obs.shape[1]
    # binary forecast

    cross = []; val = []; xl = []
    cross2 = []; val2 = []
    nvals = []
    for j in slim:

        thres = var[:] >= j
        accf = []; accp = []; ns = []
        rmsef = []; rmsep = []

        for i in range(0,horizon):
            ind = np.isfinite(obs[:,i]) & np.isfinite(fcst)[:,i] & \
                np.isfinite(ref[:,i]) & thres
            ns.append(np.sum(ind)) # number of instances
            xbin = obs[ind,i] >= lim
            ybin = fcst[ind,i] >= lim
            accf.append(accuracy(ybin,xbin,taxis=0))
            ybin = ref[ind,i] >= lim
            accp.append(accuracy(ybin,xbin,taxis=0))
            rmsef.append(err.rmse(obs[ind,i],fcst[ind,i]))
            rmsep.append(err.rmse(obs[ind,i],ref[ind,i]))


        x0 = interpol(accf,10)
        y0 = interpol(accp,10)
        ind = np.argwhere((x0-y0)>0)

        if len(ind) > 0:
            val.append(np.nanmin(accf))
            cross.append(ind[0]/10)
            xl.append(j)
        else:
            val.append(np.nanmin(accf))
            cross.append(np.nan)
            xl.append(j)

        x0 = interpol(rmsep,10)
        y0 = interpol(rmsef,10)
        ind = np.argwhere((x0-y0)>0)
        if len(ind) > 0:
            val2.append(np.nanmax(rmsef))
            cross2.append(ind[0]/10)
        else:
            val2.append(np.nanmax(rmsef))
            cross2.append(np.nan)
        #nvals.append(np.nanmin(ns))
        nvals.append(np.sum(thres))


    f, ax = subplots(figsize=(10,5))
    ax.set_title(plttitle, fontsize="x-large", fontweight="bold")

    ax2 = ax.twinx()
    ax2.grid('off')
    if plot_n:
        nvals = np.array(nvals) / nvals[0]
        l3 = ax2.plot(xl,nvals, label="# instances", linewidth=2.5, \
            linestyle='--', color='k')
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax2.set_ylabel('number of instances')
    else:
        l3 = ax2.plot(xl,val, label="minimum accuracy", linewidth=2.5, \
            linestyle=lsty[3], color='k')
        ax2.set_ylim(0,slim[-1])
        ax2.set_ylabel("accuracy", fontsize="large", fontweight="bold")



    l1 = ax.plot(xl,np.array(cross)+1,label="accuracy (forecast) > accuracy (persistence)", \
        linewidth=2.5, linestyle=lsty[1], color=lcs[1], marker=msty[1], markersize=5)
    l2 = ax.plot(xl,np.array(cross2)+1,label="rmse (forecast) < rmse (persistence)", \
        linewidth=2.5, linestyle=lsty[0], color=lcs[0], marker=msty[1], markersize=5)

    ax.set_xlabel("variability $\sigma_{k^*}$", fontsize="large", fontweight="bold")
    ax.set_ylabel("forecast horizon (min)", fontsize="large", fontweight="bold")

    ax.set_xlim(0,max(slim))
    ax.set_ylim(1,15)




    # added these three lines
    lns = l1+l2+l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=1, fontsize="large")
    subplots_adjust(left=0.08,right=0.92, bottom=0.15)




def draw_hist2d(ax,x,y, plttitle="", param_name="", maxl=None,
                color=True, lang="eng", units="$(Wm^{-2})$"):
    """
    Draw a 2d histogram / scatter density plot
    """
    from matplotlib.colors import LogNorm

    if color:
        cm = "gist_heat_r"
        cm = "viridis"
    else:
        cm = "gray_r"

    if lang == "eng":
        xlabel = "measured " + param_name; ylabel="modeled " + param_name
    elif lang == "ger":
        xlabel = "Messung " + param_name; ylabel="Modell " + param_name

    if maxl is None:
        maxl = int(np.nanpercentile(np.concatenate((x,y)),99))
        maxl = maxl + 0.3*maxl

    minl = 0

    #nbins = int( (maxl-minl) / 10. )
    nbins = 50
    meanobs = np.nanmean(x)
    e = calc_errs(x,y)
    r = e['corr']
    rmse = e['rmse']
    bias = e['mbe']
    counts,_,_,hist = ax.hist2d(x,y, bins=[nbins,nbins], \
        range=[[minl,maxl],[minl,maxl]], cmap=cm, norm=LogNorm())
    hist.set_clim(int(max([1,0.05*(maxl-minl)**2/(nbins**2)])), \
        int(np.percentile(counts[counts>0],98)))

    sc = colorbar(hist,ax=ax,pad=.05, aspect=20, shrink=0.9)
    sc.set_label('frequency')
    ax.plot(np.arange(0,maxl),np.arange(0,maxl),'k--',lw=2.0)
    ax.set_title(plttitle, fontweight="bold", fontsize='x-large')
    ax.set_xlabel(xlabel +' '+ units, fontsize='x-large')
    ax.set_ylabel(ylabel + ' ' + units, fontsize='x-large')
    ax.grid("on")
    ax.text(0.02,0.98,('# data points = %d\nRMSE = %.2f (%.1f %%)\nBias = %.2f (%.1f %%)\nCorr = %.3f' % \
        (np.sum(np.isfinite(y)), rmse, 100*rmse/meanobs, bias, \
        100*bias/meanobs, r)), fontsize='large', verticalalignment="top", \
        horizontalalignment="left", transform=ax.transAxes, \
        bbox=dict(facecolor='white', alpha=0.3))




def draw_hist2d_diff(ax,x,y,plttitle="", param_name="", maxl=None, color=True, \
    xlabel="", ylabel="", lang="eng", units=""):
    """
    Draw a 2d histogram / scatter plot where x and y have different ranges
    """

    from matplotlib.colors import LogNorm

    if color:
        cm = "viridis_r"
    else:
        cm = "gray_r"

    maxlx = np.nanmax(x) + 0.3 * np.nanmax(x)
    maxly = np.nanmax(y) + 0.3 * np.nanmax(y)

    minl = 0

    nbinsx = len(x) / 1000.
    nbinsy = len(y) / 1000.

    #meanobs = np.nanmean(x)
    #r = err.pearson(x,y)
    #rmse = err.rmse(x,y)
    #bias = err.mbe(x,y)
    counts ,_,_,hist = ax.hist2d(x,y, bins=[nbinsx,nbinsy],\
            range=[[minl,maxlx],[minl,maxly]] ,cmap=cm,norm=LogNorm())
    hist.set_clim(int(max([1,0.01*(maxlx-minl)**2/(nbinsx**2)])),
                  int(np.percentile(counts[counts>0],99)))
    sc = colorbar(hist,ax=ax,pad=.05, aspect=20, shrink=0.9)
    sc.set_label('Frequency')
    ax.plot(np.arange(0,maxl),np.arange(0,maxl),'k--',lw=2.0)
    ax.set_title(plttitle, fontweight="bold", fontsize='x-large')
    ax.set_xlabel(xlabel +' '+ param_name, fontsize='x-large')
    ax.set_ylabel(ylabel + ' ' + param_name, fontsize='x-large')
    ax.grid("on")
   # ax.text(0.02,0.98,('# Datenpunkte = %d\nRMSE = %.2f (%.1f %%)\nBias = %.2f (%.1f %%)\nCorr = %.3f' % (np.sum(np.isfinite(y)), rmse, 100*rmse/meanobs, bias, 100*bias/meanobs, r)),\
    #    fontsize='large', verticalalignment="top",horizontalalignment="left",transform=ax.transAxes)







def plot_cmv(ax, dates, data, par="Speed", slc=slice(None,-1,None),
             plttitle="", xlabel="", sampling="", color=True):

    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty

    # Datetime for x-axis
    dates = dates

    u = data['cmv_u'][slc]; v = data['cmv_v'][slc]
    us = data['cmv_spd_sigma'][slc]; vs = np.degrees(data['cmv_dir_sigma'][slc])

    sd = dates[0].replace(hour=0,minute=0,second=0)
    ed = (dates[-1] + timedelta(days=1)).replace(hour=0,minute=0,second=0)

    # Date / Xtick Labeling
    Locator = AutoDateLocator()
    dateform = AutoDateFormatter(Locator)
    dateform.scaled[1/(24.)] = '%H'

    # resampling
    if sampling != "":
        u = Series(u, index=dates).resample(sampling, label="right").mean()
        v = Series(v, index=dates).resample(sampling, label="right").mean()
        vs = Series(us, index=dates).resample(sampling, label="right").mean()
        us = Series(vs, index=dates).resample(sampling, label="right").mean()
        dates = u.index.to_pydatetime()

    if np.sum(np.isfinite(u)) == 0:
        print("No valid data given")
        return

    if par=="Speed":

        if plttitle: ax.set_title(plttitle, fontsize="medium", fontweight="bold")
        # Speed
        speed = np.sqrt(u**2 + v**2)
        upper_speed = speed + us
        lower_speed = speed - us

        # Speed plot
        if plttitle: title(plttitle)
        ax.plot(dates,speed,lw=1,label='Speed',color=lcs[0])
        fill_between(dates,lower_speed, upper_speed, ax = ax, alpha=0.5,
                     linewidth=0, color=lcs[0])
        ax.set_ylabel('Cloud Speed ($ms^{-1}$)', fontsize="small")
        ax.xaxis.set_major_locator(Locator)
        ax.xaxis.set_major_formatter(dateform)
        ax.set_ylim(0,30)

    elif par == "Direction":

        if plttitle: ax.set_title(plttitle, fontsize="medium", fontweight="bold")
        # Direction
        direc = np.arctan2(u,v)
        direc[direc<0] = direc[direc<0] + 2*np.pi
        direc = direc % (2*np.pi)
        direc = np.degrees(direc)

        upper_direc = direc + vs
        lower_direc = direc - vs

        # Direction plot
        ax.plot(dates, direc, lw=1, label='Direction', color=lcs[1])
        fill_between(dates, lower_direc, upper_direc, alpha=0.5, linewidth=0,
                     facecolor=lcs[1])
        ax.set_ylabel(u'Cloud Direction (°)', fontsize="small")
        ax.set_ylim(0,360)
        ax.xaxis.set_major_formatter(dateform)
        ax.xaxis.set_major_locator(Locator)

    ax.set_xlabel(xlabel)
    ax.set_xlim(sd,ed)



def WindRose(data):

    from windrose import WindroseAxes

    cnt = 0

    fig = figure(2,figsize=(10,10),facecolor='w', edgecolor='k')

    u = data['cmv_u'][:-1]; v = data['cmv_v'][:-1]
    us = data['cmv_spd_sigma'][:-1]; vs = np.degrees(data['cmv_dir_sigma'][:-1])
    # Speed
    speed = np.sqrt(u**2 + v**2)
    # Direction
    direc = np.arctan2(u,v)
    direc[direc<0] = direc[direc<0] + 2*np.pi
    direc = direc % (2*np.pi)
    direc = np.degrees(direc)


    ws = speed
    wd = direc

    rect = [0.1,0.1,0.8, 0.8]
    ax = WindroseAxes(fig,rect,axisbg='w')

    fig.add_axes(ax)

    ax.contourf(wd, ws, bins=np.arange(0,20,2), cmap=cm.YlGnBu)
    ax.contour(wd, ws, bins=np.arange(0,20,2), colors="black")
    cnt += 1
    l = ax.legend( loc="upper left",bbox_to_anchor = (0.8, 0.45), shadow=True)
    setp(l.get_texts(), fontsize=12.)
    title = 'Cloud motion\nRiem 2015'
    fig.text(0.03,0.95,title,verticalalignment="top",horizontalalignment="left",fontsize=24.,fontweight="bold")


def plot_cloudcover_hist(data):

    ax = subplot()
    ax.hist(data, bins=100, range=(0,100), normed=True)
    ax.set_xlabel('Cloud Coverage %')
    ax.set_ylabel('Frequency')


def plot_cloudcover_tseries(ax, dates, data):

    from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    ax.set_title("Cloud cover fish eye", fontsize="medium", fontweight="bold")
    ax.plot(dates, data, lw=2.0, label="Fish eye")
    #ax.set_xlabel(get_tlabel())
    ax.set_ylabel("CC [%]")

    mean = np.nanmean(data)
    ax.text(0.02,0.95,('Average %.1f %%' % (mean)), \
        fontsize=15., horizontalalignment="left", \
        verticalalignment="top",transform=ax.transAxes)

    # Date / Xtick Labeling
    Locator = AutoDateLocator()
    formatter = AutoDateFormatter(Locator)
    formatter.scaled[1/(24.)] = '%H'
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(Locator)
    ax.set_ylim(0,105)




def plot_sza(ax, dates, data):

    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    ax.plot(dates, data, lw=2.0)
    #ax.set_xlabel(get_tlabel())
    ax.set_ylabel("SZA [°]")

    # Date / Xtick Labeling
    Locator = AutoDateLocator()
    dateform = AutoDateFormatter(Locator)
    dateform.scaled[1/(24.)] = '%H'

    ax.xaxis.set_major_formatter(dateform)
    ax.xaxis.set_major_locator(Locator)
    ax.set_ylim(0,90)


def eyield_err(ax, x, y, p, step=step, label1="", label2="", color=True, method="rmse"):
    """
    Plot energy yield forecast RMSE vs forecast horizon.
    Example: for horizon 15 the 15 (min) average is computed for
    both forecast and measurement, then....
    """

    if color:
        lcs = c_lcs; lsty = c_lsty; msty = c_msty
    else:
        lcs = g_lcs; lsty = g_lsty; msty = g_msty

    err_yield = []; err_yield_p = []; xs = []
    # Forecast horizon
    horizon = x.shape[1]
    # Average of measurements for normalization
    meanobs = np.nanmean(x[:,0])

    # Compute error metrics for the forecast averages / yield forecasts
    for lead in range(1,horizon+1):
        x1 = np.nanmean(x[:,:lead],axis=1)
        p1 = np.nanmean(p[:,:lead],axis=1)
        y1 = np.nanmean(y[:,:lead],axis=1)
        ind = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(p1)

        if method == "rmse":
            err_yield.append(err.rmse(x1[ind],y1[ind]))
            err_yield_p.append(err.rmse(x1[ind],p1[ind]))
        elif method == "mae":
            err_yield.append(err.mae(x1[ind],y1[ind]))
            err_yield_p.append(err.mae(x1[ind],p1[ind]))
        xs.append(lead)

    ax.plot(xs,err_yield, ls=lsty[0], color=lcs[0], lw=2, label=label1)
    ax.plot(xs,err_yield_p, ls=lsty[1], color=lcs[1], lw=2, label=label2)
    ax2 = ax.twinx()
    ax1Ys = ax.get_yticks()
    ax2Ys = []
    for Y in ax1Ys:
        ax2Ys.append( "%d" % (100. * Y / meanobs) )
    ax2.set_yticks(ax1Ys); ax2.set_ybound(ax.get_ybound()); ax2.set_yticklabels(ax2Ys)
    ax2.set_ylabel("rel. "+method.upper() + ' (%)', fontsize="small", fontweight="bold")

    ax.set_ylabel(method.upper() + ' ($Wm^{-2}$)', fontsize="small", fontweight="bold")
    ax.set_xlabel("Forecast Horizon / Averaging time",
                  fontsize="small", fontweight="bold")
    ax.legend(fancybox=True, loc="lower right", fontsize="x-small")
