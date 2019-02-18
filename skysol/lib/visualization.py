# Import MatplotLib for visualization
import matplotlib.pyplot as plt
import time
from datetime import datetime
import cv2
import os
import numpy as np
from skysol.lib import optical_flow, misc, drawings
from numpy import degrees, radians, arctan2, pi
from matplotlib.dates import date2num, DateFormatter, DayLocator, HourLocator, MinuteLocator
from matplotlib.ticker import MaxNLocator, LinearLocator
from scipy.ndimage.interpolation import rotate
from PIL import Image
import cmocean

#===============================================================================
#
# Matplotlib settings
#
#===============================================================================
plt.rcParams['ytick.labelsize'] = 11.
plt.rcParams['xtick.labelsize'] = 11.
plt.rcParams['axes.labelcolor'] = '000000'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 12.
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = 'ffffff'
plt.rcParams['xtick.major.size' ] = 5.5      # major tick size in points
plt.rcParams['xtick.minor.size' ] = 3.5      # major tick size in points
plt.rcParams['ytick.major.size' ] = 5.5      # major tick size in points
plt.rcParams['ytick.minor.size' ] = 3.5      # major tick size in points
plt.rcParams['ytick.major.width' ] = 2      # major tick size in points
plt.rcParams['xtick.major.width' ] = 2      # major tick size in points
plt.rcParams['ytick.color'] = '000000'
plt.rcParams['xtick.color'] = '000000'
plt.rcParams['grid.color']  = 'black'   # grid color
plt.rcParams['grid.linestyle'] =  ':'       # dotted
plt.rcParams['grid.linewidth'] = 0.2     # in points
plt.rcParams['font.size'] = 11.
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['legend.fontsize'] = 'small'
plt.rc('mathtext', fontset='cm', default='regular')

def patch_image_cache(style, cache_dir='tilecache'):
    """
    Monkey patch the ``get_image()`` method of ``tiles`` to read and write image
    tiles from ``cache_dir`` before trying to download them.
    """
    from cartopy.io.img_tiles import GoogleTiles

    tiles = GoogleTiles(style=style)
    # Ensure cache directory exists.
    os.makedirs(cache_dir, exist_ok=True)
    def get_image(tile):
        cache_img = os.path.join(cache_dir, style + '_%d_%d_%d.png' % tile )
        if os.path.exists(cache_img):
            img = Image.open(cache_img).convert(tiles.desired_tile_form)
            return img, tiles.tileextent(tile), 'lower'
        # Call get_image() method of tiles instance and store the downloaded image.
        img, extent, origin = type(tiles).get_image(tiles, tile)
        img.save(cache_img, 'PNG')
        return img, extent, origin
    tiles.get_image = get_image
    return tiles

def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else plt.gca()
    ax.fill_between(x, y1, y2, **kwargs)
    p = plt.Rectangle((0, 0), 0, 0)#, **kwargs)
    ax.add_patch(p)
    return p



def scale_bar(ax, lat, lon, length, location=(0.5, 0.05), linewidth=5):
    """
    ax is the axes to draw the scalebar on.
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    """
    import utm
    import cartopy.crs as ccrs
    #Projection in metres, need to change this to suit your own figure
    zone = utm.latlon_to_zone_number(lat,lon)
    if lat < 0:
        sh = True
    else:
        sh = False
    utm_c = ccrs.UTM(zone, southern_hemisphere=sh)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm_c)
    #Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    #Generate the x coordinate for the ends of the scalebar
    for i in range(0,length):
        if i % 2 == 0:
            c = 'k'
        else:
            c = 'w'
        bar_xs = [sbcx - length * 500 + i * 1000, sbcx - length * 500 + (i+1) * 1000]
        #Plot the scalebar
        ax.plot(bar_xs, [sbcy, sbcy], transform=utm_c, color=c, linewidth=linewidth)
    #Plot the scalebar label
    sbcy = sbcy + (y1 - y0) * 0.02
    ax.text(sbcx, sbcy, str(length) + ' km', color="black", transform=utm_c, fontweight="bold",
            horizontalalignment='center', verticalalignment='bottom', fontsize=15)



def plot(outfile, in_img, actdate, nstations, pyr, csk, ini,cmv, \
    xsun, ysun, mask, csl, cmap, features, hist_flag=False, text_flag=False,
    params=None):

    fig = plt.figure(figsize=(16,9))

    ncols = 5; nrows = 3

    # get station index
    if ini.fcst_flag and nstations > 0:
        k = [j for j in range(0, nstations) if int(pyr[j].ind) == int(ini.statlist[0])][0]
    else:
        k = 0

    # Cloud classification
    if ini.cloud_class_apply:
        CC_long = ['Cumulus','Cirrus','Altocumulus','Clear Sky','Stratocumulus', 'Stratus', 'Nimbostratus']
        CC_short = ['Cu','Ci/Cs','Ac/Cc','Clear','Sc', 'St', 'Ns/Cb']
        ccstr_long = CC_long[params['imgclass']-1]
        ccstr_short = CC_short[params['imgclass']-1]
        if meta['imgclass'] > 0:
            cpstr = str(np.round(params['imgprob'][params['imgclass']-1],2))
        else:
            cpstr = "-1"
    else:
        ccstr_long = ""
        ccstr_short = ""
        cpstr = ""

    img = cmap.copy()

    if ini.cbh_flag:

        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

        #-------------------------------------------------------------------
        # create map
        #-------------------------------------------------------------------

        style = "satellite"
        # load OSM background image
        background = patch_image_cache(style, \
            ini.rootdir + '/tmp')

        ax = plt.subplot2grid((nrows,ncols), (0,2), \
            colspan=2, rowspan=2, projection=background.crs)

        # set boundaries of map
        ax.set_extent((ini.lon_min, ini.lon_max, ini.lat_min, ini.lat_max ))
        bnd = ax.get_extent()

        # Add the background to the map
        res = ini.x_res * ini.grid_size
        if res > 10000:
            ax.add_image(background,12,alpha=0.9)
        elif res > 5000 and res <= 10000:
            ax.add_image(background,13,alpha=0.9)
        else:
            ax.add_image(background,14,alpha=0.9)

        #ax.imshow(background)
        gl = ax.gridlines(draw_labels=True,
                      linewidth=1, color='white', alpha=0.6, linestyle='--')
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        # draw cloud/shadow map
        ax.imshow(img,cmap=plt.cm.gray,alpha=0.5, \
            zorder=1, vmin=0, transform=background.crs, origin="upper",\
            extent=bnd)

        # Draw a scale bar
        scale_bar(ax, ini.lat0, ini.lon0, 5, linewidth=10)

        # Mark camera position
        sct = ax.scatter(ini.lon0, ini.lat0, \
            s=25, marker='x',c="red", transform=background.crs.as_geodetic())

    else:

        # draw cloud map
        ax = plt.subplot2grid((nrows,ncols), (0,2), colspan=2, rowspan=2)
        sct = ax.imshow(img, vmin=0, cmap=plt.cm.get_cmap('RdBu_r'))
        ax.grid('off')
        plt.title('Irradiance Map')
        plt.axis('off')


    # Forecast arrow
    if ini.flow_flag:

        # Point forecast
        xvals = []; yvals = []; vals = []
        cm = plt.cm.get_cmap('RdBu_r')
        cm = cmocean.cm.solar
        # Draw forecast arrow
        if ini.draw_forecast_path:
            for i in range(0, ini.fcst_horizon):
                inind = int(i / ini.fcst_res)
                x = int(pyr[k].fpos[i][1])
                y = int(pyr[k].fpos[i][0])
                if x > cmap.shape[0] - 2 or x <= 0 or y <= 0 or y > cmap.shape[1]-2: continue
                xvals.append(x); yvals.append(y)
                cskval = csk.ghi[csk.tind]
                vals.append(pyr[k].fghi[inind])

            if ini.cbh_flag:
                xvals = np.array(xvals)[np.isfinite(vals)]
                yvals = np.array(yvals)[np.isfinite(vals)]
                vals = np.array(vals)[np.isfinite(vals)]
                lats, lons = misc.grid2latlon(ini.lat0,ini.lon0,ini.x_res, ini.y_res, ini.grid_size, xvals, yvals)
                if len(xvals) > 0:
                    sct = ax.scatter(lons, lats, s=30, vmin=0.15 * cskval,
                           vmax=cskval + 0.15 * cskval, marker='o', c=vals, cmap=cm, \
                           edgecolor='none', transform=background.crs.as_geodetic(),zorder=10)
                # Draw station dots
                sct2 = plot_stat(ax, ini, nstations, pyr, csk.ghi[csk.tind], k,
                    transform=background.crs.as_geodetic())
            else:

                sct = ax.scatter(xvals, yvals, s=30, vmin=0.15 * csk.ghi[csk.tind],
                           vmax=csk.ghi[csk.tind] + 0.15 * csk.ghi[csk.tind], marker='o', c=vals, cmap=cm,
                           edgecolor='none')

            # Colorbar
            try:
                cbar = plt.colorbar(mappable=sct, pad=.02, aspect=18, shrink=0.85)
            except ( AttributeError, TypeError, UnboundLocalError ):
                pass

    # Select area to cut from image
    imgsize = in_img.orig_color.shape
    x0 = int(ini.cy-ini.fx)
    if x0 < 0: x0 = 0
    x1 = int(ini.cy+ini.fx)
    if x1 > imgsize[0]: x1 = imgsize[0]
    y0 = int(ini.cx-ini.fy)
    if y0 < 0: y0 = 0
    y1 = int(ini.cx+ini.fy)
    if y1 > imgsize[1]: y1 = imgsize[1]


    # Origin Image
    plt.subplot2grid((nrows,ncols), (0,0), colspan=1, rowspan=1)
    img = in_img.orig_color_draw.copy()
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #img = rotate(img[x0:x1,y0:y1],-np.degrees(ini.rot_angles[2]))
    cv2.circle(img,(ysun,xsun),15,0,-1)
    img = img[x0:x1,y0:y1]
    plt.axis('off')
    plt.imshow(img)
    plt.title('Original Image')
    del img

    # RBR
    ax = plt.subplot2grid((nrows,ncols), (1,1))
    img = in_img.rbr.copy() * 1.0
    img[mask] = np.nan
    img = img[x0:x1,y0:y1]
    a = ax.imshow(img,vmin=ini.rbr_thres-0.2, vmax=ini.rbr_thres+0.2,cmap=plt.cm.viridis)
    cbar = plt.colorbar(a,pad=.03,aspect=15,shrink=0.7, format="%.2f" )
    plt.axis('off')
    if csl == 0: plt.title('RBR')
    if csl == 1: plt.title('RBR - CSL')
    if csl == 2: plt.title('RBR corrected')



    if hist_flag:

        in_img.rbr[mask]=np.nan
        plt.subplot2grid((nrows,ncols), (2,0),colspan=1)
        plt.hist((in_img.rbr.flatten()), \
            range=(0.3,1.3), bins=125, color="red",alpha=0.5,normed=True)
        plt.ylim(0,15)
        plt.axvline(ini.rbr_thres, color='b', linestyle='dashed', linewidth=2)
        plt.legend(['RBR threshold','RBR'],loc=2)

        if ini.csi_mode == "hist" and ini.radiation:
            ind = pyr[k].tind
            y = np.divide( pyr[k].ghi[ind-ini.avg_csiminmax:ind], csk.ghi[csk.tind-ini.avg_csiminmax:csk.tind] )
            y = y[np.isfinite(y)]
            if len(y) > (0.6*ini.avg_csiminmax):
                ax = plt.subplot2grid((nrows,ncols), (2,1),colspan=1)
                plt.hist((y),bins=ini.hist_bins, color="red",range=(0.0,1.5))
                plt.axvline(pyr[k].csi_min, color='b', linestyle='dashed', linewidth=2)
                plt.axvline(pyr[k].csi_max, color='b', linestyle='dashed', linewidth=2)
                plt.xlim(0.2,1.5)
                ax.text(0.2,1.05,'k* histogram',fontsize=9,transform=ax.transAxes)

    # Clear Sky Reference
    if csl == 1:
        plt.subplot2grid((nrows,ncols), (1,0))
        img = in_img.cslimage
        img[mask] = np.nan
        img = img[x0:x1,y0:y1]
        a = plt.imshow(img,vmin=0.5, vmax=1.2,cmap=plt.cm.viridis)
        plt.title('CSL')
        plt.axis('off')
        plt.colorbar(a,pad=.03, aspect=15,shrink=0.7)

    if ini.plot_features:

        for f in range(0,len(features.vec)):
            if f > len(features.vec)/2:
                xo = 0.7; yo = 0.3-(f-len(features.vec)/2)/50.
            else:
                xo = 0.43; yo = 0.3-f/50.
            txt = '%g' % (features.vec[f])
            fig.text(xo,yo,features.names[f][0:26])
            fig.text(xo+0.17,yo,txt)

    # RBR differences
#     img = in_img.rbr_orig - in_img.rbr
#     plt.subplot2grid((nrows,ncols), (1,0))
#     img[mask] = np.nan
#     a = plt.imshow(img[x0:x1,y0:y1],cmap=plt.cm.get_cmap('bwr'),vmin=-0.2,vmax=0.2)
#     plt.axis('off')
#     plt.colorbar(a,pad=.03, aspect=15,shrink=0.7)
#     plt.title('RBR diff')

    # Binary cloud mask
    plt.subplot2grid((nrows,ncols), (0,1))
    img = in_img.binary_color.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img[in_img.mask_horizon] = 0
    img = img[x0:x1,y0:y1]
    #img = rotate(img,np.degrees(ini.rot_angles[2]))
    plt.title('Cloud decision')
    plt.axis('off')
    plt.imshow(img)



    # Draw timeseries
    past = ini.plot_last_vals
    horizon = int(ini.fcst_horizon/ini.fcst_res) + 1
    horizon = int(ini.fcst_horizon)
    if hist_flag:
        ax = plt.subplot2grid((nrows,ncols), (2,2),colspan=3)
    elif ini.plot_features:
        ax = plt.subplot2grid((nrows,ncols), (2,0),colspan=2)
    else:
        ax = plt.subplot2grid((nrows,ncols), (2,0),colspan=5)

    maxval = 0

    i = 0

    if ini.radiation:

        # Plot measurements
        if ini.live:
            slc = slice(pyr[k].tind-past,pyr[k].tind,ini.fcst_res)
            x = pyr[k].time[slc]
            y = pyr[k].ghi[slc]
            y2 = pyr[k].dhi[slc]
        else:
            slc = slice(pyr[k].tind-past,pyr[k].tind+horizon,ini.fcst_res)
            x = pyr[k].time[slc]
            y = pyr[k].ghi[slc]
            y2 = pyr[k].dhi[slc]
        dates=[datetime.utcfromtimestamp(ts) for ts in x ]
        if len(dates) > 0: plt.plot(dates, y, 'b-',lw=2.0, label="Measurement")
        if len(y2) > 0:
            fill_between(dates,0,y2,alpha=0.5,linewidth=0,facecolor="yellow", label="DHI")
            fill_between(dates,y2,y,alpha=0.5,linewidth=0,facecolor="orange", label="DNI")

    # Analysis Values
    nvals = ini.plot_last_vals / ini.camera_res / ini.rate
    x = pyr[k].aghi_time[-int(nvals):]
    dates=[datetime.utcfromtimestamp(ts) for ts in x if ~np.isnan(ts) ]
    if len(dates) > 0:
        y = pyr[k].aghi[-len(dates):]
        plt.plot(dates, y, 'gv', label="Analysis")

    # Clear sky irradiance
    slc = slice(csk.tind-ini.plot_last_vals, csk.tind+ini.fcst_horizon, ini.fcst_res)
    x = csk.time[slc]
    dates=[datetime.utcfromtimestamp(ts) for ts in x ]
    y = csk.ghi[slc]
    plt.plot(dates, y, '--', color='black', label="Clear Sky")
    maxval = 1.7 * csk.actval
    plt.ylabel('Irradiance in $Wm^{-2}$')

    # Forecast Values
    x = pyr[k].ftime
    dates=[ datetime.utcfromtimestamp(ts) for ts in x if ~np.isnan(ts) ]
    y = pyr[k].fghi[:len(dates)]

    plt.plot(dates,y,'r-',lw=2.0, label="Forecast")

    # Vertical line to plot current time instance
    plt.axvline(actdate, color='b', linestyle='--', lw=2.0)
    plt.xlabel('Time [UTC]')
    plt.legend(loc="upper left", ncol=3, fontsize=8)
    plt.ylim([0,maxval])

    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(LinearLocator(numticks=6))
    ax.xaxis_date()

    # Draw Text
    ax = plt.subplot2grid((nrows,ncols), (0,4),rowspan=2)
    ax.axis('off')
    nowtime = datetime.strftime(datetime.utcnow(),"%Y-%m-%d %H:%M:%S")
    acttime = datetime.strftime(actdate,"%Y-%m-%d %H:%M:%S")
    loctime = str(in_img.locdate.isoformat(' '))
    ax.text(-0.3,0.95,acttime + str(' UTC'), weight="bold")
    ax.text(0.2,0.02,'Created:\n' + nowtime + str(' UTC'),fontsize=9)
    ax.text(-0.3,0.9,"Sun Zenith = " + str(round(params['sza'],1)) + '$^\circ$' )
    ax.text(-0.3,0.86,"Sun Azimuth = " + str(round(params['saz'],1)) + '$^\circ$' )
    if ini.cbh_flag: ax.text(-0.3,0.82,'Cloud Base Height: ' + \
        str(int(params['cbh'])) + ' m ')
    ax.text(-0.3,0.79,'Cloud Type: ' + ccstr_long + ' ' + ccstr_short + ' ' + cpstr)
    ax.text(-0.3,0.72,'Radiation measurements \n' + params['txt'] + ':' )
    ax.text(-0.3,0.65,"GHI = " + str(round(params['ghi'],1)) + ' $W/m^2$ (' + str(round(params['csi_ghi'],2))+')' )
    ax.text(-0.3,0.61,"DHI = " + str(round(params['dhi'],1)) + ' $W/m^2$ (' + str(round(params['csi_dhi'],2))+')' )
    ax.text(-0.3,0.57,"DNI = " + str(round(params['dni'],1)) + ' $W/m^2$ (' + str(round(params['csi_dni'],2))+')' )

    if ini.mode <= 1:
        ax.text(-0.3,0.40,'Cloud Cover =  '  + str(round(params['cc'],1)) + ' %' )


    if ini.flow_flag:
        if ini.cbh_flag:
            unit = "m/s"
        else:
            unit = "pix/s"
        ax.text(-0.3, 0.34, '#CMV =  ' + str(np.sum(cmv.flag)))
        um = cmv.speed[-1]; vm = cmv.direction[-1]
        ume = cmv.sspeed[-1]; vme = cmv.sdirection[-1]
        ax.text(-0.3,0.30,'All speed = '  + str(round(um,2)) + '$\pm$' + str(round(ume,2)) + unit)
        ax.text(-0.3,0.26,'All direction =  '  + str(round(np.degrees(vm),2)) + '$\pm$' + str(round(np.degrees(vme),2)) +'$^\circ$')
        um = cmv.mean_speed; vm = cmv.mean_direction
        ume = cmv.std_speed; vme = cmv.std_direction
        ax.text(-0.3,0.22,'Global speed = '  + str(round(um,2)) + '$\pm$' + str(round(ume,2)) + unit)
        ax.text(-0.3,0.18,'Global direction =  '  + str(round(np.degrees(vm),2)) + '$\pm$' + str(round(np.degrees(vme),2)) +'$^\circ$')

    ax.text(-0.3,0.14,'Lens Clear =  '  + str(params['img_qc']))
    if in_img.useful_image:
        qc = "OK"
    else:
        qc = "BAD"
    ax.text(-0.3,0.10,'Quality Flag =  ' + qc)


    # Final settings
    fig.set_figwidth = 16.
    fig.set_figheight = 9.
    fig.set_dpi = 50.
    fig.subplots_adjust(hspace=0.15,wspace=0.4,left=0.05, right=0.97, top=0.95, bottom=0.08)
    plt.savefig(outfile,format=ini.outformat)
    plt.clf()
    plt.close('all')





def plot_stat(ax, ini, nstations, pyr, cskval, k, transform=None):
    cm = plt.cm.get_cmap('RdBu_r')
    cm = cmocean.cm.solar
    xsct = []; ysct = []; val = []; flag = []; isdata = []
    # collect data from single stations
    for i in range(0, nstations):
        x = float(pyr[i].map_y)
        y = float(pyr[i].map_x)
        z = np.nanmean(pyr[i].ghi[pyr[i].tind-ini.rate:pyr[i].tind])
        xsct.append(x)
        ysct.append(y)
        val.append(z)
        if np.isfinite(z):
            isdata.append(True)
            flag.append(pyr[i].qflag[pyr[i].tind])
        else:
            isdata.append(False)
            flag.append(-1)
    isdata = np.array(isdata)
    xsct = np.array(xsct)
    ysct = np.array(ysct)
    val = np.array(val)
    # geographical coordinates
    if transform is not None:
        if np.sum(isdata) == 0:
            lats, lons = misc.grid2latlon(ini.lat0,ini.lon0,ini.x_res, ini.y_res, \
                ini.grid_size, xsct, ysct)
            sct = ax.scatter(lons, lats, \
                s=25, marker='x',c="red", transform=transform)
        else:
            lats, lons = misc.grid2latlon(ini.lat0,ini.lon0,ini.x_res, ini.y_res, \
                ini.grid_size, np.array(xsct), np.array(ysct))
            sct = ax.scatter(lons[isdata], lats[isdata], s=25, marker='x', vmin=0.15 * cskval, \
                vmax=cskval + 0.15 * cskval, c=val[isdata], cmap=cm, edgecolor='none', \
                transform=transform, zorder=30)
    else:
        # Use grid coordinates
        sct = ax.scatter(xsct[k], ysct[k], s=25, marker='o', vmin=0.15 * cskval, \
            vmax=cskval + 0.15 * cskval, c=val[k], cmap=cm, edgecolor='none')

    return sct


def plot_detection_full(outfile, ini, in_img, mask, **params):

    ncols = 5; nrows = 2
    textsize=19

    # Select area to cut from image
    imgsize = in_img.orig_color.shape
    x0 = int(ini.cy-ini.fx)
    if x0 < 0: x0 = 0
    x1 = int(ini.cy+ini.fx)
    if x1 > imgsize[0]: x1 = imgsize[0]
    y0 = int(ini.cx-ini.fy)
    if y0 < 0: y0 = 0
    y1 = int(ini.cx+ini.fy)
    if y1 > imgsize[1]: y1 = imgsize[1]


    fig = plt.figure(figsize=(15,6))


    # Original Image
    ax = plt.subplot2grid((nrows,ncols), (0,0))
    img = in_img.orig_color.copy()
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = img[x0:x1,y0:y1]
    plt.axis('off')
    ax.text(0.03,0.85,'a)',color="white",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
    plt.imshow(img)


    # Pixel Intensity
    ax = plt.subplot2grid((nrows,ncols), (1,0))
    img = 1.0 * in_img.orig_gray.copy()
    img[mask] = np.nan
    img = img[x0:x1,y0:y1]
    a = plt.imshow(img, vmin=0, vmax=255, cmap=plt.cm.viridis)
    cb = plt.colorbar(a,shrink=0.7,aspect=15)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    ax.text(0.03,0.9,'b)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
    plt.axis('off')


    # CSL Image
    ax = plt.subplot2grid((nrows,ncols), (0,1))
    try:
        img = in_img.cslimage
        img[mask] = np.nan
        a = plt.imshow(img[x0:x1,y0:y1],vmin=0.5, vmax=1.2, cmap=plt.cm.viridis)
        ax.text(0.03,0.9,'c)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
        plt.axis('off')
        cb = plt.colorbar(a,shrink=0.7,aspect=15)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(12)
    except AttributeError:
        pass

    # RBR Original
    ax = plt.subplot2grid((nrows,ncols), (0,2))
    img = in_img.rbr_orig
    img[mask] = np.nan
    a = plt.imshow(img[x0:x1,y0:y1],vmin=0.5, vmax=1.2, cmap=plt.cm.viridis)
    ax.text(0.03,0.9,'e)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
    plt.axis('off')
    cb = plt.colorbar(a,pad=.03, aspect=15,shrink=0.7)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)

    # RBR diff
    img = in_img.rbr_orig - in_img.rbr
    ax = plt.subplot2grid((nrows,ncols), (1,1))
    img[mask] = np.nan
    a = plt.imshow(img[x0:x1,y0:y1],cmap=plt.cm.get_cmap('bwr'),vmin=-0.5,vmax=0.5)
    plt.axis('off')
    ax.text(0.03,0.9,'d)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
    cb = plt.colorbar(a,pad=.03, aspect=15,shrink=0.7)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)



     # RBR modified
    ax = plt.subplot2grid((nrows,ncols), (1,2))
    img = in_img.rbr.copy() * 1.0
    img[mask] = np.nan
    img = img[x0:x1,y0:y1]
    a = ax.imshow(img,vmin=ini.rbr_thres-0.2, vmax=ini.rbr_thres+0.2, cmap=plt.cm.viridis)
    cb = plt.colorbar(a,pad=.03,aspect=15,shrink=0.7)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    ax.text(0.03,0.9,'f)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
    plt.axis('off')

    # Cloud map original
    sky_bool = in_img.rbr_orig <= ini.rbr_thres
    cloud_bool = in_img.rbr_orig > ini.rbr_thres
    binary_color = in_img.orig_color.copy()
    binary_color[sky_bool, :] = [ 255, 0 , 0 ]
    binary_color[cloud_bool, :] = [ 255, 255 , 255 ]
    binary_color[mask, :] = 0 # mask

    ax = plt.subplot2grid((nrows,ncols), (0,3))
    img = binary_color.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img[x0:x1,y0:y1]
    plt.axis('off')
    ax.imshow(img)
    ax.text(0.03,0.85,'g)',color="white",fontweight="bold",fontsize=textsize,transform=ax.transAxes)


    # Cloud map corrected
    ax = plt.subplot2grid((nrows,ncols), (1,3))
    img = in_img.binary_color.copy()
    #cloud_bool = (img != [0,0,0]) & (img != [255,0,0])
    #img[cloud_bool] = 255
    #if ini.mask_sun: img[in_img.sun_mask] = 0
    #if ini.dyn_horizon: img[in_img.sun_mask] = 0
    img[mask] = 0
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img[x0:x1,y0:y1]
    plt.axis('off')
    ax.imshow(img)
    ax.text(0.03,0.85,'h)',color="white",fontweight="bold",fontsize=textsize,transform=ax.transAxes)

    # Histogram
    ax = plt.subplot2grid((nrows,ncols), (0,4))
    in_img.rbr_orig[mask] = np.nan
    ax.hist(( in_img.rbr_orig.flatten()) , range=(0.0,1.1), bins=256, normed=True, color="blue")
    ax.set_ylim(0,15)
    plt.axvline(ini.rbr_thres, color='b', linestyle='dashed', linewidth=2)
    ax.text(0.03,0.85,'i)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)


    ax = plt.subplot2grid((nrows,ncols), (1,4))
    in_img.rbr[mask] = np.nan
    plt.hist((in_img.rbr.flatten()), range=(0.0,1.1), bins=256, color="blue",normed=True)
    plt.ylim(0,15)
    plt.axvline(ini.rbr_thres, color='b', linestyle='dashed', linewidth=2)
    ax.text(0.03,0.85,'j)',color="black",fontweight="bold",fontsize=textsize,transform=ax.transAxes)


    # Final settings
    fig.subplots_adjust(hspace=0.1,wspace=0.2,left=0.01, right=0.99, top=0.99, bottom=0.05)
    plt.savefig(outfile,format=ini.outformat)
    plt.clf()



def plot_detection(outfile, ini, in_img):
    """ Plot only raw image and binary decision """
    ncols = 2; nrows = 1
    textsize=19

    # Select area to cut from image
    imgsize = in_img.orig_color.shape
    x0 = int(ini.cy-ini.fx)
    if x0 < 0: x0 = 0
    x1 = int(ini.cy+ini.fx)
    if x1 > imgsize[0]: x1 = imgsize[0]
    y0 = int(ini.cx-ini.fy)
    if y0 < 0: x0 = 0
    y1 = int(ini.cx+ini.fy)
    if y1 > imgsize[1]: y1 = imgsize[1]

    fig = plt.figure(figsize=(6,3))

    # Original Image
    ax = plt.subplot2grid((nrows,ncols), (0,0))
    img = in_img.orig_color.copy()
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = img[x0:x1,y0:y1]
    plt.axis('off')
    ax.text(0.03,0.85,'a)',color="white",fontweight="bold",fontsize=textsize,transform=ax.transAxes)
    ax.imshow(img)


    # Cloud map
    ax = plt.subplot2grid((nrows,ncols), (0,1))
    img = in_img.binary_color.copy()
    cloud_bool = (img != [0,0,0]) & (img != [255,0,0])
    img[cloud_bool] = 255
    #if ini.mask_sun: img[in_img.sun_mask] = 0
    #if ini.dyn_horizon: img[in_img.sun_mask] = 0
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img[x0:x1,y0:y1]
    plt.axis('off')
    ax.imshow(img)
    ax.text(0.03,0.85,'b)',color="white",fontweight="bold",fontsize=textsize,transform=ax.transAxes)

    # Final settings
    fig.subplots_adjust(hspace=0.1,wspace=0.2,left=0.01, right=0.99, top=0.99, bottom=0.05)
    plt.savefig(outfile,format=ini.outformat)
    plt.clf()



def plot_paper_juelich(outfile,in_img,actdate,nstations,pyr,csk,ini,cmv, \
    xsun,ysun, mask,csl,\
    cmap,features,hist_flag=False,text_flag=False,**params):

    plt.close('all')

    fig = plt.figure(1,figsize=(9,9),facecolor='w', edgecolor='k')

    if hist_flag:
        ncols = 3; nrows = 3
    else:
        ncols = 3; nrows = 3



    st = time.time()


    # get station index
    if not ini.cbh_flag and ini.fcst_flag:
        k = [j for j in range(0, nstations) if int(pyr[j].ind) == int(ini.statlist[0])][0]
    else:
        k = 0




    # Select area to cut from image
    imgsize = in_img.orig_color.shape
    x0 = int(ini.cy-ini.fx)
    if x0 < 0: x0 = 0
    x1 = int(ini.cy+ini.fx)
    if x1 > imgsize[0]: x1 = imgsize[0]
    y0 = int(ini.cx-ini.fy)
    if y0 < 0: x0 = 0
    y1 = int(ini.cx+ini.fy)
    if y1 > imgsize[1]: y1 = imgsize[1]




    # Origin Image
    plt.subplot2grid((nrows,ncols), (0,0))
    img = in_img.orig_color_draw.copy()
    img[mask] = 0
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #img[xsun-15:xsun+15,ysun-15:ysun+15,:] = 0 # mark sun position
    img = img[x0:x1,y0:y1]
    img = rotate(img,-np.degrees(ini.rot_angles[2]))
    i1 = np.min(np.where(img!=0)[0])
    i2 = np.max(np.where(img!=0)[0])
    j1 = np.min(np.where(img!=0)[1])
    j2 = np.max(np.where(img!=0)[1])

    img = img[i1:i2,j1:j2]
    #img = np.float32(img)
    #img[img<=0]=np.nan

    plt.axis('off')
    plt.imshow(img)
    plt.title('Masked original image')
    del img



    plt.subplot2grid((nrows,ncols), (1,0))
    img = in_img.binary_color.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img[x0:x1,y0:y1]
    img = rotate(img,-np.degrees(ini.rot_angles[2]))
    #img = np.float32(img)
    #img[img<=0]=np.nan
    i1 = np.min(np.where(img!=0)[0])
    i2 = np.max(np.where(img!=0)[0])
    j1 = np.min(np.where(img!=0)[1])
    j2 = np.max(np.where(img!=0)[1])
    img = img[i1:i2,j1:j2]
    plt.title('Cloud decision map')
    plt.axis('off')
    plt.imshow(img)






    ax = plt.subplot2grid((3,3), (0,1), colspan=2, rowspan=2)

    xvals = []; yvals = []; vals = []

    # Draw shadow map
    cm = plt.cm.get_cmap('jet')
    img = cv2.cvtColor(cmap,cv2.COLOR_RGB2GRAY) * 1.0
    img[cmap[:,:,2]==0] = np.nan
    img[cmap[:,:,2]==255] = 200
    img[(cmap[:,:,2]>1)&(cmap[:,:,2]<255)]=100

    for i in range(0, ini.fcst_horizon):

        x = int(pyr[k].fpos[i][1])
        y = int(pyr[k].fpos[i][0])
        if x > cmap.shape[0] - 2 or x <= 0 or y <= 0 or y > cmap.shape[1] - 2: continue
        xvals.append(x); yvals.append(y)
        vals.append(pyr[k].fghi[i])
        #if pyr[k].fghi[i] > 500:
        #    cmap[x, y] = 200
        #else:
        #    cmap[x, y] = 90
    ax1 = plt.scatter(xvals, yvals, s=30, vmin=0.15 * csk.actval, vmax=csk.actval + 0.15*csk.actval, marker='o', c=vals, cmap=cm, edgecolor='none')
    plot_stat(ax, ini, nstations, pyr, csk.ghi[csk.tind], k)
    arr = cv2.imread(ini.rootdir + '/config/arrow.png')
    if arr is not None:
        arr[arr==64]=255;arr[arr==68]=255;arr[arr==0]=255
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        ofs = int(img.shape[0] / 150)
        arr = cv2.resize(arr,(arr.shape[0]/2,arr.shape[1]/2))
        img[ofs:ofs+arr.shape[0],ofs:ofs+arr.shape[1]] = arr
    # Title
    tit = 'Cloud base height: ' + str(params['cbh'])

    plt.title(tit, fontsize=12)
    ax.imshow(img,alpha=0.6,cmap=plt.cm.gray)


    # Rename ticks
    s = (ini.grid_size * ini.x_res) / 2.
    nticks = len(ax.get_xticks()) - 1
    steps = 2.*s / (nticks-1)
    new_ticks = list(map(int,np.arange(-s-steps,s+(2*steps),steps)))
    ax.xaxis.set_ticklabels(new_ticks,fontsize=10)
    ax.yaxis.set_ticklabels(new_ticks, fontsize=10)
    #ax.set_ylabel('latitudinal distance from camera [m]',fontsize=13)
    ax.set_xlabel('longitudinal distance from camera [m]',fontsize=13)


    # Draw timeseries
    ax = plt.subplot2grid((nrows,ncols), (2,0),colspan=3)

    maxval = 0

    i = 0

    # Forecast Values
    x = csk.time[csk.tind:csk.tind+ini.fcst_horizon]
    dates=[datetime.utcfromtimestamp(ts) for ts in x ]
    y = pyr[k].fghi[:]
    plt.plot_date(x=dates,y=y,fmt='r-',lw=2.0)

    if ini.radiation:

        # Analyses Values
        slc = slice(csk.tind-1800,csk.tind,ini.camera_res)
        x = csk.time[slc]
        dates=[datetime.utcfromtimestamp(ts) for ts in x ]
        y = pyr[k].aghi
        plt.plot_date(x=dates,y=y,fmt='gv')
        # Plot measurements
        x = pyr[k].time[ pyr[k].tind-1800:pyr[k].tind+ini.fcst_horizon]
        y = pyr[k].ghi[ pyr[k].tind-1800:pyr[k].tind+ini.fcst_horizon]
        y2 = pyr[k].dhi[ pyr[k].tind-1800:pyr[k].tind+ini.fcst_horizon]
        dates=[datetime.utcfromtimestamp(ts) for ts in x ]
        plt.plot_date(x=dates, y=y, fmt='b-',lw=2.0)
        if len(y2)>0:
            p = fill_between(dates,0,y2,alpha=0.5,linewidth=0,facecolor="yellow")
            p = fill_between(dates,y2,y,alpha=0.5,linewidth=0,facecolor="orange")

    # Clear sky irradiance
    x = csk.time[csk.tind-1800:csk.tind+ini.fcst_horizon]

    dates=[datetime.utcfromtimestamp(ts) for ts in x ]
    y = csk.ghi[csk.tind-1800:csk.tind+ini.fcst_horizon]
    plt.plot_date(x=dates,y=y,fmt='--',color='black')
    maxval = 1.5 * csk.actval

    plt.axvline(dates[1800], color='b', linestyle='--',lw=2.0)

    plt.xlabel('Time [UTC]',fontsize=14)
    plt.ylabel('Irradiance [W/m$^2$]',fontsize=14)
    if ini.radiation:
        plt.legend(['Forecast','Analysis','Measurement','Clear Sky'])
    else:
        plt.legend(['Forecast','Clear Sky'])

    plt.ylim([0,maxval])

    if ini.location == "juelich":
        plt.title('Station #' + str(pyr[k].ind)  )
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.grid('off')
    #ax.xaxis.set_major_locator(HourLocator())
    ax.xaxis_date()


    # Draw Text
    acttime = datetime.strftime(datetime.utcnow(),"%Y-%m-%d %H:%M:%S")
    plt.text(0.7,0.95,str(np.char.capitalize(ini.location)) + ' \n' + \
        actdate + str(' UTC'), weight="bold")#, fontsize=22. )


    # Final settings
    fig.subplots_adjust(hspace=0.4,wspace=0.3,left=0.085, right=0.95, top=0.95, bottom=0.08)
    plt.draw()
    plt.savefig(outfile,dpi=200)
    plt.clf()



def plot_original_mod(img, outfile, mask, dt, ini, cam, title="Sky Imager", **meta):

    from PIL import Image, ImageFont, ImageDraw
    from skysol.lib.drawings import draw_boundaries, sunpath

    data = img.orig_color_draw.copy()

    data = sunpath(data, dt, ini, cam)
    data[mask,:] = 0

    try:
        data = cv2.resize(data, img.segments.shape)
        data = np.array(255*draw_boundaries(data, img.segments), dtype="uint8")
        data = rotate(data,-np.degrees(ini.rot_angles[2]))
    except:
        data = cv2.resize(data, (600,600))
        data = data[:,:,::-1]
        data = rotate(data,-np.degrees(ini.rot_angles[2]))

    i1 = np.min(np.where(data!=0)[0])
    i2 = np.max(np.where(data!=0)[0])
    j1 = np.min(np.where(data!=0)[1])
    j2 = np.max(np.where(data!=0)[1])
    data = data[i1:i2,j1:j2]
    #data = data[::-1,:,:]

    # PIL part
    im = Image.fromarray(data,'RGB')

    draw = ImageDraw.Draw(im)
    lx, ly = im.size

    datestr = datetime.strftime(img.locdate, "%Y-%m-%d")
    timestr = datetime.strftime(img.locdate, "%H:%M:%S")
    #txtfont = ImageFont.truetype("data/arial.ttf", 22)
    draw.text((10,10), title,fill='red')

    #txtfont = ImageFont.truetype("data/arial.ttf", 20)
    draw.text((lx-120,10), datestr,fill='red', align="right")
    draw.text((lx-120,30), timestr, fill = 'red', align="right")

    #txtfont = ImageFont.truetype("data/arial.ttf", 18)
    draw.text((10,ly-60), "Sun Elev. = %.1f°" % (90-meta['sza']), fill = 'red', align="right")
    draw.text((10,ly-40), "Sun Azimuth = %.1f°" % meta['saz'], fill = 'red', align="right")
    draw.text((10,ly-20), "Cloud Cover = %.1f%%" % meta['cc'], fill = 'red', align="right")

    if meta['img_qc']:
        qf = "OK"
    else:
        qf = "BAD"

    draw.text((lx-100,ly-60), "QC = %s" % qf, fill = 'red', align="right")
    CC_long = ['Cumulus','Cirrus','Altocumulus','Clear Sky','Stratocumulus', 'Stratus', 'Nimbostratus']
    CC_short = ['Cu','Ci/Cs','Ac/Cc','Clear','Sc', 'St', 'Ns/Cb']
    ccstr_long = CC_long[meta['imgclass']-1]
    ccstr_short = CC_short[meta['imgclass']-1]
    if meta['imgclass'] > 0:
        cpstr = str(np.round(meta['imgprob'][meta['imgclass']-1],2))
    else:
        cpstr = "-1"
    draw.text((lx-120,ly-40), "%s (%s%%)" % (ccstr_short, cpstr), fill = 'red',  align="right")

    im.save(outfile)




def plot_cmv(outfile, in_img, ini, cam, cmv,
        xsun, ysun, mask, map_x, map_y, **params):
    """
    Time for example: 2013-04-25 14:15:30

    Shi-Tomasi Corner Detection parameter:
    self.feature_params = dict( minDistance = 50,
                           blockSize = 12)
    self.maxCorners = 500
    self.qualityLevel = 0.03
    """

    plt.close('all')

    # Figure size
    fig = plt.figure(1,figsize=(8,8),facecolor='w', edgecolor='k')

    # Image Subplot order
    nrows = 2; ncols = 2

    # Original Image
    ax = plt.subplot2grid((nrows,ncols), (0,0))
    img = in_img.orig_color_draw.copy()
    img[cmv.cmv_mask,:] = [0,0,255]
    img[mask] = 0
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #img = img[x0:x1,y0:y1]
    cv2.circle(img,(ysun,xsun),20,[250,200,25],-1)
    #img = rotate(img,-np.degrees(ini.rot_angles[2]))
    i1 = np.min(np.where(img!=0)[0])
    i2 = np.max(np.where(img!=0)[0])
    j1 = np.min(np.where(img!=0)[1])
    j2 = np.max(np.where(img!=0)[1])
    img = img[i1:i2,j1:j2]
    plt.axis('off')
    plt.imshow(img)
    del img
    ax.text(0.03,0.95,'Fisheye RGB Image',color="white",fontweight="bold",fontsize=12,transform=ax.transAxes)


    # Segmented cloud/sky image with CMV
    ax = plt.subplot2grid((nrows,ncols), (0,1))
    img = in_img.orig_gray.copy()
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img[cmv.cmv_mask,:] = [255,0,0]
    # Draw every 50th cloud path
    if len(cmv.x) > 0: drawings.cloud_path(img, cmv.x, cmv.y, lcolor=[25,25,125])
    img[mask] = 0
    img[in_img.mask_horizon] = 0
    cv2.circle(img,(ysun,xsun),20,[250,200,25],-1)
    #img = rotate(img,-np.degrees(ini.rot_angles[2]))
    img = img[i1:i2,j1:j2]
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.viridis)
    ax.text(0.03,0.95,'Fisheye Intensity',color="white",fontweight="bold",fontsize=12,transform=ax.transAxes)

    del img

    # Projected RGB with CMV
    ax = plt.subplot2grid((nrows,ncols), (1,0))
    img = in_img.orig_color_draw.copy()
    img[cmv.cmv_mask,:] = [0,0,255]
    img = cam.grid(img, map_x, map_y)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.axis('off')
    plt.imshow(img)
    ax.text(0.03,0.95,'Perspective projection RGB',color="white",fontweight="bold",fontsize=12,transform=ax.transAxes)

    # Projected binary with CMV
    ax = plt.subplot2grid((nrows,ncols), (1,1))
    img = in_img.binary_color
    img[cmv.cmv_mask,:] = [0,0,255]
    if len(cmv.x) > 0: drawings.cloud_path(img, cmv.x, cmv.y, lcolor=[25,25,125])
    img = cam.grid(img, map_x, map_y)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.axis('off')
    plt.imshow(img)
    ax.text(0.03,0.95,'Perspective projection cloud map',color="white",
        fontweight="bold",fontsize=12,transform=ax.transAxes)

    fig.text(0.5,0.5, "#CMV %d - Direction %.1f° - Speed %.1f" % (np.sum(cmv.flag),np.degrees(cmv.mean_direction), \
    cmv.mean_speed), fontsize=15, horizontalalignment='center')


    # Final settings
    plt.tight_layout()#pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.draw()
    plt.savefig(outfile,dpi=100)
    plt.clf()
    plt.close('all')
