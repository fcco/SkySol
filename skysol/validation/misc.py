import numpy as np
from datetime import datetime
import getopt, sys, os
#import configobj as cfo
#from configobj import ConfigObjError, flatten_errors
import numpy as np
from skysol.validation import error_metrics as err
import pytz

def argHandler(argv,cfg):

    print(cfg)
    # define defaults
    params = {}
    bdir = os.path.dirname(sys.argv[0])
    if bdir == "": bdir = "."
    params['inpath'] = cfg['Paths']['HDF5_PATH']
    params['date'] = datetime.utcnow()
    params['sdate'] = None
    params['edate'] = None
    params['outpath'] = None
    params['quiet'] = False
    params['station'] = int(cfg['Misc']['station'])
    params['step'] = int(cfg['Misc']['step'])
    params['horizon'] = int(cfg['Misc']['horizon'])
    params['meas_path'] = cfg['Paths']['Meas_PATH']

    USAGE = "USAGE:\n" \
        "python validate.py\n\n" \
        "\t-h\t this help\n"\
        "\t--quiet          \t suppress output on stdout\n"\
        "\t-d / --date=     \t string, input date as YYYYMMDD\n"\
        "\t-p / --datapath=  \t string, path to hdf5 folder\n"\
        "\t-o / --outpath=  \t string, path for all output\n"\
        "\t-s / --start_date=  \t string, start date as YYYYMMDD\n"\
        "\t-e / --end_date=  \t string, start date as YYYYMMDD\n"
    # read arguments

    try:
        opts, args = getopt.getopt(argv[1:], "h:d:p:o:s:e:", \
        ["date=", "sdate==", "edate==", "h5path=", "outpath=", "quiet"])
    except getopt.GetoptError as er:
        print("\n" + str(er) + "\n\n" + USAGE)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ("-d", "--date"):
            params['date'] = datetime.strptime(str(arg),"%Y%m%d")
        elif opt in ("-s", "--start_date"):
            params['sdate'] = datetime.strptime(str(arg),"%Y%m%d")
        elif opt in ("-e", "--end_date"):
            params['edate'] = datetime.strptime(str(arg),"%Y%m%d")
        elif opt in ("-p", "--h5path"):
            params['inpath'] = str(arg)
        elif opt in ("-o", "--outpath"):
            params['outpath'] = str(arg)
        elif opt in ("--quiet"):
            params['quiet'] = True
    return params



def dty(par):
    if par == "forecast":
        return np.float16
    elif par == "Datetime" or par == "time" or par == "Time" or par == "Timestamp":
        return np.int64
    elif par == "Timestring":
        return np.str
    else:
        return np.float32

def read_niwe(filename):
    import pandas as pd
    tzi = pytz.timezone("Asia/Colombo")

    # Read CSV line by line an read the last column
    with open(filename) as f:
        lis=[line.split(',') for line in f]        # create a list of lists
    ts = [l[0] for l in lis[1:]]

    # Convert strings with timestamps to python internal datetime objects
    dts = [ datetime.strptime(d, "%d-%m-%Y %H:%M:%S") for d in ts ]
    # Convert local datetime to UTC datetime to make it compatible with image timestamps
    ldt = [ tzi.localize(dt,is_dst=False).astimezone(pytz.UTC) for dt in dts ]

    # Measured GHI
    ghi = [ int(l[-2]) for l in lis[1:]]
    # Unix Timestamp
    time = [ d.timestamp() for d in ldt ]
    # Datetime objects
    dates = np.array([ pytz.UTC.localize(datetime.utcfromtimestamp(ts)) for ts in time ])

    # interpolate measurements to 1s resolution
    ghi = pd.Series(ghi, index = dates)
    res = ghi.resample('1S').interpolate()
    dates = [ts.to_pydatetime() for ts in res.index]
    time = [ dt.timestamp() for dt in dates ]
    ghi = np.array(res.values)
    data = {}
    data['time'] = time
    data['dates'] = dates
    data['ghi'] = ghi

    return data

def read_hdf5(params,inpath='./',fmt="%Y%m%d",exclude=[], include=[], horizon=None, timekey="Datetime"):

    import h5py

    if params['sdate'] is not None and params['edate'] is not None:
        days = date_range(params['sdate'],params['edate'])
    else:
        days = [ params['date'] ]

    data = {}; cnt = 0
    for day in days:
        dtstring = datetime.strftime(day,fmt)
        file =  inpath + '/' + dtstring
        try:
            with h5py.File(file,'r') as f:
                if cnt == 0:

                    for key in f.keys():
                        if type(f[key]) == h5py._hl.group.Group:
                            if key in exclude: continue
                            for dset in f[key].keys():
                                if dset in exclude: continue
                                if (include != []) and (key not in include): continue
                                data[key + '/' + dset.lower()] = np.array(f[key][dset],dtype=dty(key))
                        else:
                            key
                            if key in exclude: continue
                            data[key.lower()] = np.array(f[key][:],dtype=dty(key))
                else:

                    for key in f.keys():
                        if type(f[key]) == h5py._hl.group.Group:
                            if key in exclude: continue
                            for dset in f[key].keys():
                                if dset in exclude: continue
                                if (include != []) and (key not in include): continue
                                try:
                                    data[key + '/' + dset.lower()] = np.concatenate((data[key + '/' + dset.lower()], \
                                        np.array(f[key][dset])), axis=0 )
                                except KeyError:
                                    data[key + '/' + dset.lower()] = np.array(f[key][dset],dtype=dty(key))
                        else:
                            if key in exclude: continue
                            if (include != []) and (key not in include): continue
                            data[key.lower()] = np.concatenate((data[key.lower()], np.array(f[key][:],dtype=dty(key))), axis=0)

                try:
                    cnt = cnt + len(f['Time'])
                except KeyError:
                    cnt = cnt + len(f[timekey])

        except OSError as e:
            print(e,'File ' + file + ' not Found! - Skip!')
            continue

        if timekey != "Time":
               data['time'] = np.array(data[timekey.lower()],dtype=np.int64)

    return data


def loadConfig(inpath):
    """
    Load the configuration from 'settings.ini' and validates against 'settings_spec.ini'.
    """
    import sys
    if inpath == "": inpath = "."
    # Load Configuration
    try:
        cfg = cfo.ConfigObj(inpath + os.sep + 'settings.ini',configspec=inpath + os.sep + 'settings_spec.ini', file_error=True)
    except (ConfigObjError, IOError) as  e:
        sys.exit('Could not read "%s": %s' % ("settings_spec.ini", e))
    return cfg


def date_range(start_date, end_date):
    import datetime
    """
    Returns a generator of all the days between two date objects.

    Results include the start and end dates.

    Arguments can be either datetime.datetime or date type objects.

    h3. Example usage

        >>> import datetime
        >>> import calculate
        >>> dr = calculate.date_range(datetime.date(2009,1,1), datetime.date(2009,1,3))
        >>> dr
        <generator object="object" at="at">
        >>> list(dr)
        [datetime.date(2009, 1, 1), datetime.date(2009, 1, 2), datetime.date(2009, 1, 3)]

    """
    # If a datetime object gets passed in,
    # change it to a date so we can do comparisons.
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime.datetime):
        end_date = end_date.date()

    # Verify that the start_date comes after the end_date.
    if start_date > end_date:
        raise ValueError('You provided a start_date that comes after the end_date.')

    # Jump forward from the start_date...
    while True:
        yield start_date
        # ... one day at a time ...
        start_date = start_date + datetime.timedelta(days=1)
        # ... until you reach the end date.
        if start_date > end_date:
            break


# Function for smoothing
def smooth(x, window_len=10, window='hanning'):

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')

    return y[window_len-1:-window_len+1]


def list_duplicates(datain):
    seq = list(datain)
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set( x for x in seq if x in seen or seen_add(x) )
    # turn the set into a list (as requested)
    ind = np.zeros_like(datain, dtype=np.bool)
    ind[:] = True
    for i in sorted(list(seen_twice)):
        b = np.where(i==datain)[0][:]
        for c in range(0,len(b)-1):
            ind[b[c]] = False
    diff = datain[ind][1:] - datain[ind][:-1]
    notinc = np.where(diff<0)[0]
    if len(notinc) > 0:
        ind[:notinc[-1]] = False
    return ind





def windAverage(spd,wdir,intime,window):
        """
        Calculate wind averages.

        Input:

        spd =  vector of wind speed values
        wdir = vector of wind direction values
        intime = vector of time instances
        window = lenght of time window for averaging

        Output:

        mean_speed = average wind speed in time window (dtype=np.ndarray)
        max_speed = maximum wind speed value in time window (gusts) (dtype=np.ndarray)
        mean_direction = average wind direction in time window (dtype=np.ndarray)
        time_arr = time instances according to time window (dtype=np.ndarray)

        """
        mean_speed =[]; mean_direction = []; max_speed = []; time_arr = []

        # calculate components first
        u = spd * np.sin(wdir * np.pi/180.)
        v = spd * np.cos(wdir * np.pi/180.)

        # loop over time instances with discrete time steps
        for i in range(0,len(spd),window):

            time_arr.append(intime[i:i+window][-1])
            avg_u = np.nanmean(u[i:i+window])
            avg_v = np.nanmean(v[i:i+window])
            # vector of wind speeds in time window
            speed = np.sqrt(u[i:i+window]**2 + v[i:i+window]**2)

            # average speed in time window
            mean_speed.append(np.nanmean(speed))
            # average direction in time window
            y = np.arctan2(avg_u,avg_v)
            if y < 0: y += 2*np.pi
            y = y % (2*np.pi)
            mean_direction.append(np.degrees(y))
            # maximum of windspeed in time window
            max_speed.append(np.nanmax(speed))

        return np.array(mean_speed), np.array(max_speed), np.array(mean_direction), np.array(time_arr)



def u_vs_v(t, x, y, pe, c, step_size=1, window_size=90, horizon=300):

    n_iter = int(x.shape[0] / float(step_size))

    Ue = []; Ve = []; Upe = []; ts = []


    # Variability
    tmp = x.copy()
    tmp[:,0] = pe[:,0]

    for i in range(0,n_iter):

        slc = slice(i*step_size,(i*step_size) + int(window_size))
        length = slc.stop - slc.start

        if np.sum(np.isfinite(y[slc,horizon])) > 0.2*length:

            # Uncertainty Forecast
            Ue.append(err.U(x[slc,horizon],y[slc,horizon],c[slc,horizon]))

            # Uncertainty Persistence
            Upe.append(err.U(x[slc,horizon],pe[slc,horizon],c[slc,horizon]))

            vari, k = err.V(tmp[slc,:],c[slc,:],t=horizon)
            Ve.append(vari)

            ts.append(t[slc][0])

    return ts, Ue, Upe, Ve


def get_persistence(obs,cls,l=10):
    csi = np.divide(obs[:,0],cls[:,0])
    p = np.array([csi,]*l).T
    return p


def calc_errs(x,y,p=None,c=None,t=0,cmin=50.,taxis=-1,min_val=0.2):
    """
    Calculate error metrics RMSE, MBE, MAE, Correlation, FS
        and return them in a dictionary

    Skill scores are only calculated if a reference forecast is given

    :param x: vector of observations
    :param y: vector if forecasts
    :param p: optional, vector of reference forecasts (e.g. persistence)
    :param c: vector of clear sky irradiance to retrieve normalized values
    :param t: optional, default 0; timelag for variability calculation in indizes
    :param cmin: optional, default 50;  minimum values of clear sky reference to be used in
          the calculations
    :param taxis: optional, default -1; calculate error metric along given axis
    :param min_val: optional, default 0: minimum percentage of values that must be finite in order to compute error metrics (range 0-100)
    :return errs: dictionary with error metrics ('rmse','mbe','corr','mae','fs','nvalid')

    """
    errs = {}
    if x.ndim == 1:
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        if p is not None:
            p = p[:,np.newaxis]

    if taxis >= 0:
        errs['rmse'] = err.rmse(x,y,taxis=taxis)
        errs['mbe'] = err.mbe(x,y,taxis=taxis)
        errs['corr'] = err.vcorrcoef(x,y,taxis=taxis)
        errs['mae'] = err.mae(x,y,taxis=taxis)
        errs['nvalid'] = np.isfinite(y)
        if not(p is None): errs['fs'] = err.FS(x,y,p,taxis=taxis)
        if not(p is None) and not(c is None): errs['sscore'] = err.sscore(x,y,c,t,cmin,taxis=taxis)
    else:
        errs['rmse'] = err.rmse(x,y)
        errs['mbe'] = err.mbe(x,y)
        errs['corr'] = err.vcorrcoef(x,y)
        errs['mae'] = err.mae(x,y)
        errs['nvalid'] = np.isfinite(y)
        if not(p is None): errs['fs'] = err.FS(x,y,p)
        if not(p is None) and not(c is None):
            errs['sscore'] = err.sscore(x,y,c,t,cmin)


    if min_val > 0:

        if taxis >= 0:
            min_ind = np.sum(np.isfinite(y), axis=taxis) / y.shape[taxis] <= min_val
            errs['mbe'][min_ind] = np.nan
            errs['rmse'][min_ind] = np.nan
            errs['mae'][min_ind] = np.nan
            errs['corr'][min_ind] = np.nan
            if not(p is None) and not(c is None):
                errs['sscore'][min_ind] = np.nan
            if not(p is None): errs['fs'][min_ind] = np.nan

            if taxis == 0:
                errs['nvalid'][:,min_ind] = False
            if taxis == 1:
                errs['nvalid'][min_ind,:] = False
        else:
            min_ind = np.sum(np.isfinite(y)) / np.size(y) <= min_val
            if min_ind:
                return np.nan

    return errs


def accuracy(y_true,y_pred,taxis=0):
    """
    Accuracy score for binary forecasts
    """
    TP = np.sum((y_pred == True) & (y_true == True),axis=taxis)
    TN = np.sum((y_pred == False) & (y_true == False),axis=taxis)
    FP = np.sum((y_pred == True) & (y_true == False),axis=taxis)
    FN = np.sum((y_pred == False) & (y_true == True),axis=taxis)

    return np.divide( (TP+TN) ,(TP+FP+FN+TN).astype(float) )


def variability(data, axis = 0):
    return np.nanstd(data,axis=0)

def interpol(data,x):
    """
    Resamples data by given factor with interpolation
    """
    from scipy.interpolate import interp1d

    # Resamples data by given factor by interpolation
    x0 = np.linspace(0, len(data)-1, len(data))
    x1 = np.linspace(0, len(data)-1, len(data)*x-(x-1))
    f = interp1d(x0, data)
    return f(x1)
