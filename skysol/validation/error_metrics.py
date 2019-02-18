# encoding=utf8
from __future__ import division
import numpy as np

"""
Different error metrics. 

Defintion and description of some from 
Zhang et al., 2013,  Metrics for Evaluating the
Accuracy of Solar Power Forecasting, conference paper, 
3rd International Workshop on Integration of
Solar Power into Power Systems


"""
def ksi(fcst,obs):
        
    """
    Calculates the Kolmogorov-Smirnow Test Integral (KSI)

    The KSI and OVER metrics were proposed by Espinar et
    al. 12. The Kolmogorov–Smirnov (KS) test is a
  
    nonparametric test to determine if two data sets are
    significantly different. The KS statistic D is defined as the
    maximum value of the absolute difference between two
    cumulative distribution functions (CDFs), expressed as

    :param x: Vector of observation values
    :param y: Vector of forecast values

    :returns ksi: The KSI
    """

    m = 100.0
    nbins = 100

    cdf_x = cdf(x,nbins=nbins)
    cdf_y = cdf(y,nbins=nbins)


    # Critical value Vc
    N = len(y)
    if N < 35:
        print("Number of data points for KSI not sufficient. N=",N,"<35")
        return np.nan
    Vc = 1.63 / np.sqrt(N)

    D = np.max(cdf_x - cdf_y)
    # Observation maximum and minimum
    Pmax = np.max(x); Pmin = np.min(x)

    # Interval distance
    d = ( Pmax - Pmin ) / m

    ksi = np.sum(D)

        
        

def pearsonr(x, y):
        
    # Assume len(x) == len(y)
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(map(lambda x, y: x * y, x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den



def pearson(x, y):
    """
    Calculates Pearson Correlation Coefficient

    Description:

    Pearson’s correlation coefficient
    is a global error measure metric; a larger value of Pearson’s
    correlation coefficient indicates an improved solar
    forecasting skill.
    
    :param x: Vector of obserations 
    :param y: Vector of forecasts

    :returns: Correlation Coefficient
    """
    
                
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = np.nanmean(x
)
    avg_y = np.nanmean(y, dtype=np.float32)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    cnt = 0
    for idx in range(n):
        if np.isnan(x[idx]) or np.isnan(y[idx]): continue
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
        cnt += 1

    if cnt == 0: return np.nan
                              
    return diffprod / np.sqrt(xdiff2 * ydiff2)



def vcorrcoef(X,y,taxis=-1):
    """
    Calculates Pearson Correlation Coefficient (with axis functionality)

    Description:

    Pearson’s correlation coefficientq
    is a global error measure metric; a larger value of Pearson’s
    correlation coefficient indicates an improved solar
    forecasting skill.
    
    :param x: Vector of obserations 
    :param y: Vector of forecasts
    :param taxis (optional): Axis along which the means are computed

    :returns: Correlation Coefficient
    """
    
    ndims = X.ndim
    assert ndims < 3
    
    if taxis >= 0:
        Xm = np.nanmean(X,axis=taxis, dtype=np.float32)
        ym = np.nanmean(y,axis=taxis, dtype=np.float32)
  
        Xm = Xm.reshape(Xm.shape[0],1)
        ym = ym.reshape(ym.shape[0],1)
            
        if taxis == 0: Xm = Xm.T
        if taxis == 0: ym = ym.T
                
    else:
        Xm = np.nanmean(X, dtype=np.float32)
        ym = np.nanmean(y, dtype=np.float32)

    diffx = np.subtract(X,Xm)
    diffy = np.subtract(y,ym)
    
    prod1 = np.multiply( diffx, diffy )
    prodx = np.multiply( diffx, diffx )
    prody = np.multiply( diffy, diffy )
    
    prodx[np.isnan(prod1)] = np.nan
    prody[np.isnan(prod1)] = np.nan

    if taxis >= 0:       
        r_num = np.nansum(prod1,axis=taxis)
        r_den = np.sqrt( np.multiply(np.nansum(prodx,axis=taxis), np.nansum(prody,axis=taxis) ))
    else:    
        r_num = np.nansum(prod1)
        r_den = np.sqrt( np.nansum(prodx) * np.nansum(prody) )
        
    r = np.divide(r_num,r_den)
    
    return r



def rmse(x,y,taxis=-1):
    
    """
    Calculates root mean square error (RMSE) if 
    an observation and forecast vector are given. 
    Both vectors must have same length, so pairs of
    elements with same index are compared.

    Description: The RMSE provides a global error measure during
    the entire forecasting period.
       
    :param x: vector of observations
    :param y: vector of forecasts
    :param taxis (optional): Axis along which the means are computed

    :returns: RMSE
    """
    if taxis >= 0:
        return np.sqrt(np.nanmean( np.square( np.subtract(x,y), dtype=np.float32), axis=taxis) )
    else:
        return np.sqrt(np.nanmean( np.square( np.subtract(x,y), dtype=np.float32) ))

def maxae(x,y, taxis=-1):

    """
    Calculates maximum absolute error (MaxAE) if 
    an observation and forecast vector are given. 
    Both vectors must have same length, so pairs of
    elements with same index are compared.

    Description:

    The MaxAE is an indicative of local deviations of
    forecast errors.
    The MaxAE metric is useful to evaluate the forecasting of
    short-term extreme events in the power system.

    :param x: vector of observations
    :param y: vector of forecasts
    :param taxis (optional): Axis along which the means are computed
 
    :returns: MaxAE
    """
    if taxis >= 0:
        return np.nanmax(abs(x-y),axis=taxis,dtype=np.float32)
    else:
        return np.nanmax(abs(x-y),dtype=np.float32)


def mae(x,y,taxis=-1):

    """
    Calculate mean absolute error (MaxAE) if 
    an observation and forecast vector are given. 
    Both vectors must have same length, so pairs of
    elements with same index are compared.

    Description:
    
    The MAE has been widely used in regression problems
    and by the renewable energy industry to evaluate forecast
    performance.

    The MAE metric is also a global error measure metric, which,
    unlike the RMSE metric, does not excessively account for
    extreme forecast events.


    :param x: vector of observations
    :param y: vector of forecasts
    :param taxis (optional): Axis along which the means are computed

    :returns: MAE
    """
    if taxis >= 0:
        return np.nanmean(abs(x-y), axis=taxis,dtype=np.float32)
    else:
        return np.nanmean(abs(x-y),dtype=np.float32)

def mape(x,y,fac,taxis=-1):

    """
    Calculate mean absolute percentage error (MAPE) if 
    an observation and forecast vector are given. Additionaly
    a normalizing value must be given, e.g. capacity factor, 
    average CSI,...
    Both vectors must have same length, so pairs of
    elements with same index are compared.

    Description:

    Same as MAE but normalized differences are normalized 
    to a given value.


    :param x: vector of observations
    :param y: vector of forecasts
    :param fac: value for normalization (e.g. capacity factor, mean csi)
    :param taxis (optional): Axis along which the means are computed


    :returns: MAPE
    """

    if taxis >= 0:
        return np.nanmean(abs( (x-y)/fac ), axis=taxis,dtype=np.float32)
    else:
        return np.nanmean(abs( (x-y)/fac ) ,dtype=np.float32)
        

def mbe(x,y,taxis=-1):

    """
    Calculate mean biae error (MBE) if 
    an observation and forecast vector are given. 
    Both vectors must have same length, so pairs of
    elements with same index are compared.

    Description:

    The MBE metric intends to indicate average forecast bias.
    Understanding the overall forecast bias (over- or under-
    forecasting) would allow power system operators to better
    allocate resources for compensating forecast errors in the
    dispatch process.


    :param x: vector of observations
    :param y: vector of forecasts
    :param taxis (optional): Axis along which the means are computed


    :returns: MBE
    """
 
    if taxis >= 0:
        return np.nanmean((x-y),axis=taxis,dtype=np.float32)
    else:
        return np.nanmean(x-y,dtype=np.float32)
        
        
def FS(x,y,p,method="RMSE",taxis=0):
    """ 
    Calculates Forecast Skill (FS)
    
    FS is defined as 1 - ( Error(Forecast) / Error(Reference) )
    
    
    :param x: Vector of observation values
    :param y: Vector of forecast values
    :param p: Vector of reference forecast

    :returns: FS
    
    """
    
    err1 = rmse(x,y,taxis=taxis)
    err2 = rmse(x,p,taxis=taxis)
    
    return ( 1 - np.divide(err1,err2) )
    

def skewness(x,y):

    """
    Calculate skewness of the probability distribution 
    of the forecast error if 
    an observation and forecast vector are given. 
    Both vectors must have same length, so pairs of
    elements with same index are compared.
    
    Description:
    
    Skewness is a measure of the asymmetry of the
    probability distribution, and is the third standardized moment
    
    Assuming that
    forecast errors are equal to forecast power minus actual
    power, a positive skewness of the forecast errors leads to an
    over-forecasting tail, and a negative skewness leads to an
    under-forecasting tail. The tendency to over-forecast (or
    under-forecast) is important in that the system actions taken 
    to correct for under-forecasting and over-forecasting events
    are not equal. An over-forecasting tendency could lead to a
    less than optimal number of large thermal units being
    committed, which need to be corrected through the starting
    of more expensive, but faster starting, units in the dispatch
    process.
    
    
    
    :param x: vector of observations
    :param y: vector of forecasts
    
    
    :returns: Skewness
    """
    from scipy.stats import skew
    
    return skew(x-y)


def kurtosis(x,y):

    """
    Calculate kurtosis of the probability 
    distribution of the forecast error if 
    an observation and forecast vector are given. 
    Both vectors must have same length, so pairs of
    elements with same index are compared.

    Description:

    Kurtosis is a measure of the magnitude of the peak of the
    distribution, or, conversely, how fat-tailed the distribution is,
    and is the fourth standardized moment

    The difference between the kurtosis of a sample distribution
    and that of the normal distribution is known as the excess
    kurtosis. In the subsequent anIn [142]: U
alysis, the term kurtosis will be
    treated synonymously with excess kurtosis. A distribution
    with a positive kurtosis value is known as leptokurtic, which
    indicates a peaked distribution; whereas a negative kurtosis
    indicates a flat data distribution, known as platykurtic. The
    pronounced peaks of the leptokurtic distribution represent a
    large number of very small forecast errors


    :param x: vector of observations
    :param y: vector of forecasts


    :returns: Kurtosis
    """
    from scipy.stats import kurtosis

    return kurtosis(x-y)


def iqrdiff(x,y):

    """
    Calculates Interquartile Range Difference (IQR Diff) 
    of a two given datasets
    
    Description: (not from the paper) IQR is the difference between
    the 75th percentile and the 25th percentile. This function
    returns the difference of two IQR. 
    Input:
    
    :param x: Vector of observation values
    :param y: Vector of forecast values
    
    
    :returns: IQR
    """
    
    iqr_x = np.percentile(x,75) - np.percentile(x,25) 
    iqr_y = np.percentile(y,75) - np.percentile(y,25) 
    return iqr_x - iqr_y

def r2(y,x):

    """
    Calculates coefficient of determination R^2

    Description: R^2 is a comparison of the variance of the 
    errors to the variance of the data which is to be modeled 
    Input:

    :param x: Vector of observation values
    :param y: Vector of forecast values
       

    :returns: R^2
    """
    r2 = 1 - ( np.nanvar(y-x) / np.nanvar(x) )
    
    return r2

def V(x,cls,t=1,cmin=50.):

    """
    Calculates solar variability V as introduced in Marquez and Coimbra (2012)
    "proposed metric for evaluation of solar forecasting models"

    Description: "Solar variability V is the standard deviation of the step-changes 
    of the measured solar irradiance to that of a clear sky solar irradiance so that 
    the diurnal variability is neglected."
    
    This method can use single-dimensional obervation and clear sky vectors with 
    subsequent and temporal equidistant instances ( timeseries ). Increments are 
    calculated with an moving window along this axis.
    
    If two-dimensional vectors are provided subsequent instances must be in the second dimension. 
    Increments are calculated in the second dimension, while iterating is done on the values 
    in the first axis.
    
    Variability is then calculated as the standard deviation of all increments.

    :param x: float vector of irradiance values
    :param cls: float vector of corresponding clear sky irradiance values
    :param t: int, optional: Timelag/stepsize t in indizes for increments
    :param cmin: float, optional: minimum values of clear sky reference to be used in
          the calculations. default is 50 W/m2.
       
    :returns: deltak = vector of clear sky index increments
    :returns: V = solar variability
    """
    
    
    def slc(arr,s,e,ndims):
        """ returns the input array ´arr´ sliced from ´s´ to ´e´ 
        at the specified axis ´taxis´"""
        irange = slice(s,e)
        items = [slice(None, None, None)] * ndims
        items[taxis] = irange
        return arr[tuple(items)]
    
    nd = x.ndim
    y = cls.copy()
        
    # don't use values for low irradiance values
    y[cls<=cmin] = np.nan
    
    if nd == 1:
        # clear sky index for time t+deltat
        #csi0 = np.divide(slc(x,t,None,nd),slc(y,t,None,nd))
        csi0 = np.divide(x[t:],y[t:])

        # clear sky index for time t
        #csi1 = np.divide(slc(x,0,-t,nd),slc(y,0,-t,nd))
        csi1 = np.divide(x[0:-t],y[0:-t])

    if nd == 2:
        # clear sky index for time t+deltat
        csi0 = np.divide(x[:,t],y[:,t])
        # clear sky index for time t
        csi1 = np.divide(x[:,0],y[:,0])
    
    
    # Difference
    deltak = np.subtract(csi0,csi1)
    
    # calculate standard deviation only if number of datapoints is large enough
    if np.sum(np.isfinite(deltak)) > 5:     
        V = np.sqrt(np.nanmean(deltak**2,axis=0,dtype=np.float32))
    else:
        V = np.nan
        
    return V, deltak

def VI(x,cls,t,cmin=50.):
    """ 
    Calculates a variability index defined by Stein et al.
    "The variability index: A new and novel metric for 
    quantifying irradiance and pv output variability"
  
    Description: Solar Variability VI over a period of time is 
    calculated as the ratio of the "length" of the measured irradiance 
    plotted against time divided by the "length" of the clear sky irradiance 
    plotted against time. On a clear day, VI would be ~ 1. The same is for very overcast
    days. Higher variability (changes in time) of irradiance will lead
    to higher values of VI.
    
    
    :param x: vector if irradiance values
    :param cls: vector of clear sky reference values
    :param t: average period in minutes
    :param cmin: minimum values of clear sky reference to be used in
          the calculations. default is 50 W/m2.
          
    
    :returns: Solar irradiance variability score ( scalar ) VI
    """
    y = cls.copy()
    y[cls<=cmin] = np.nan
    sum1 = np.nansum(np.sqrt((x[1:]-x[0:-1])**2 + t**2),dtype=np.float32)
    sum2 = np.nansum(np.sqrt((y[1:]-y[0:-1])**2 + t**2),dtype=np.float32)

    VI = np.divide(sum1,sum2)
    
    return VI

def U(x,y,cls,cmin=50.,taxis=0):
    """
    Calculates "Forecast Uncertainty" as defined my Marquez and Coimbra, 2013 
    ("Proposed Metrics for Evaulation of Solar Forecasting Models")
    
    "Here we define the uncertainty as the
    standard deviation of a model forecast error divided by the esti-
    mated clear sky value of the solar irradiance over a subset time
    window of Nw data points"

    
    :param x: vector of irradiance values
    :param y: vector of irradiance forecasts
    :param cls: vector of clear sky reference values
    :param cmin: minimum values of clear sky reference to be used in
          the calculations. default is 50 W/m2.
          
    :return U: forecast uncertainty

    """
    
    return np.sqrt( np.nanmean(np.divide( np.subtract(x,y), cls )**2, axis=taxis,dtype=np.float32) )

def sscore(x,y,cls,t,cmin=50.,taxis=0):
    
    """
    Calculating a metric for evaluating solar forecast models proposed by 
    Marquez and Coimbra (2012) 
    "proposed metric for evaluation of solar forecasting models"
    
    Description: The metric sscore is calculated as the ratio of the above defined 
    forecast uncertainity U and the timeseries variability V.
    
    sscore = 1 means a perfect forecast. sscore = 0 means the variability dominates the forecast.
    By definition a persistence forecast has a sscore = 0. A negative sscore means 
    that the forecast performs worse than a persistence forecast.
    
    
    :param x: vector of irradiance values
    :param y: vector of irradiance forecasts
    :param cls: vector of clear sky reference values
    :param t: timelag for variability calculations
    :param cmin: minimum values of clear sky reference to be used in
          the calculations. default is 50 W/m2.
          
    :returns sscore:
    """
    y[cls<=cmin] = np.nan
    x[cls<=cmin] = np.nan

    return 1 - ( np.divide(U(x,y,cls,taxis=taxis), V(x,cls,t,cmin=cmin,taxis=taxis)[0]) )
    
    

def precision(y_true,y_pred,**kwargs):
    """
    Compute the precision using sklearn module sklearn.metrics.precision_score
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    The best value is 1 and the worst value is 0.
    
    Look at sklearn.metrics.precision_score for details how to use
    
    In case of binary forecasts you can use boolean arrays or just 0 or 1s.
    """
    from sklearn.metrics import precision_score
    
    return precision_score(y_true, y_pred, **kwargs)


def roc(x,y,minmax,nbins=100,taxis=-1):
    """
    Calculate Receiver Operating Curve (ROC)
    
    :param x: observation vector
    :param y: forecast vector
    :param minmax: range of thresholds, give a tupel (e.g. (0,1) in )
    :param nbins: number of bins/thresholds inside the range
    
    :returns tp,fp: returns vector of true positive TP and false positive FP 
    for the given range of thresholds
    """
    if taxis >= 0:
        shape = list(x.shape)
        wh = shape[taxis]
        shape[taxis] = nbins
        TP = np.empty(shape)
        FP = np.empty(shape)
    else:
        TP = np.empty(nbins)
        FP = np.empty(nbins)
        x = x.flatten()
        y = y.flatten()
        wh = x.shape[0]
    ra = minmax[1] - minmax[0]
    cnt = 0
    ths = np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(nbins))
    
    for th in ths:
    
        y_pred = y >= th
        y_true = x >= th
        
        TP[cnt] = np.sum((y_pred == True) & (y_true == True),axis=taxis) / float(wh)
        FP[cnt] = np.sum((y_pred == True) & (y_true == False),axis=taxis) / float(wh)
        #print th, TP[cnt], FP[cnt]
        cnt += 1
    
    return TP, FP
    
    
    
def accuracy(y_true,y_pred,taxis=0):   
     """
     Accuracy classification score:

    
     In case of binary forecasts you can use boolean arrays or just 0 or 1s.
     """
     #from sklearn.metrics import accuracy_score
     TP = np.sum((y_pred == True) & (y_true == True),axis=taxis)
     TN = np.sum((y_pred == False) & (y_true == False),axis=taxis)
     FP = np.sum((y_pred == True) & (y_true == False),axis=taxis)
     FN = np.sum((y_pred == False) & (y_true == True),axis=taxis)
     
     return np.divide( (TP + TN) , float((TP + FP + FN + TN)))
     #return accuracy_score(y_true, y_pred, **kwargs)
    
    
    
    

def prints(x, y, c, p=""):
    """ Gives a summary of error metrics
    
    :param x: observation vector
    :param y: forecast vector
    :param c: clear sky vector
    :param p: reference vector

    :returns a: a string with a number of metrics"""
    
   
    a = "Number of measurements = %d (%.2f) \n " % (x.shape[0] - np.count_nonzero(np.isnan(x)), (x.shape[0] - np.count_nonzero(np.isnan(x))) / float(x.shape[0]))
    a = a + "Number of forecasts = %d (%.2f) \n " % (y.shape[0] - np.count_nonzero(np.isnan(y)), (y.shape[0] - np.count_nonzero(np.isnan(y))) / float(y.shape[0]))
    a = a + "RMSE = %.4f \n " % rmse(x, y)
    a = a + "BIAS = %.4f \n " % mbe(x, y)
    a = a + "CORR = %.4f \n " % pearson(x, y)
    if p != "":
        a = a + "FS = %.4f \n " % FS(x, y, p)
    a = a + "MEAN OBS = %.4f (%.3f) \n " % (np.nanmean(x), np.nanmean(x / c))
    a = a + "MEAN FOR = %.4f (%.3f) \n " % (np.nanmean(y), np.nanmean(y / c))
    a = a + "MEAN CLS = %.4f \n " % np.nanmean(c)
    a = a + "SSCORE 60s = %.4f \n " % sscore(x, y, c, 60)
    
    if p != "":
        a = a + "FS = %.4f \n " % FS(x, y, p)
    a = a + "SSCORE Persistence 60s = %.4f \n " % sscore(x, p, c, 60)

    return a
