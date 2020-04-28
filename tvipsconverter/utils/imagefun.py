import numpy as np
import math
#from scipy.misc import imresize
from scipy.ndimage import median_filter, convolve
from PIL import Image
#from scipy import stats

def scale16to8(arr, min=None, max=None):
    if min is None:
        min = arr.min()
    else:
        arr[arr<min] = min
    if max is None:
        max = arr.max()
    else:
        arr[arr>max] = max

    return ((arr-min)*255//(max-min))


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gausfilter(arr, ks, sig):
    kernel = matlab_style_gauss2D(shape=(ks,ks),sigma=sig)
    return convolve(arr, kernel, mode = "constant", cval = 0)


def medfilter(arr, ks):
    return median_filter(arr, size = ks, mode = "constant", cval = 0)


def cnnfilter(arr):
    pass


def scalestd(raw, numstd=3):
    #scaling n/2 std dev around image mean

    mean = np.mean(raw)
    std = np.std(raw)

    nstd = numstd*std

    return scale16to8(raw, mean - nstd, mean + nstd)


def scalepercent(raw, percent=0.07):
    #suppresses outliers in min/max

    minval, maxval = findoutliers(raw, percent)

    return scale16to8(raw, minval, maxval)

"""
findoutliers (array, percent=0.07)

Uses returns min and max values of the array with the upper and lower percent pixel values suppressed.
"""
def findoutliers(raw, percent=0.07):
    cnts, edges = np.histogram(raw, bins=2**16)

    stats = np.zeros((2, 2**16), dtype=np.int)

    stats[0] = np.cumsum(cnts) #low
    stats[1] = np.cumsum(cnts[::-1]) #high

    thresh = stats > percent * raw.shape[0] * raw.shape[1]

    min = (np.where(thresh[0]))[0][0]
    max = 2**16 - (np.where(thresh[1]))[0][0]

    return edges[min], edges[max+1]


def suppressoutliers(raw, percent=0.07):
    min, max = findoutliers(raw, percent)

    raw[raw<min] = min
    raw[raw>max] = max

    return raw

def bin2(a, factor):
    assert len(a.shape) == 2
    #binned = imresize(a, (a.shape[0]//factor, a.shape[1]//factor))
    binned = np.array(Image.fromarray(a).resize(
        size =(a.shape[0]//factor, a.shape[1]//factor)))
        #,resample = Image.BILINEAR )
    return binned


# def bin2(a, factor):
    # '''
    # This is based on: http://scipy.org/Cookbook/Rebinning
    # It is simplified to the case of a 2D array with the same
    # binning factor in both dimensions.
    # '''
    # assert len(a.shape) == 2
    # oldshape = a.shape
    # newshape = int(a.shape[0]/factor), int(a.shape[1]/factor)
    # tmpshape = (newshape[0], factor, newshape[1], factor)
    # f = factor * factor
    # binned = np.sum(np.sum(np.reshape(a, tmpshape), 1), 2) / f
    # return binned

#def bin2m(a, factor):
#       '''
#       Median instead of mean for bin2
#       '''
#       oldshape = a.shape
#       newshape = np.asarray(oldshape)/factor
#       tmpshape = (newshape[0], factor, newshape[1], factor)
#       binned = stats.median(stats.median(np.reshape(a, tmpshape), 1), 2)
#       return binned



def getElectronWavelength(ht):
        # ht in Volts, length unit in meters
        h = 6.6e-34
        m = 9.1e-31
        charge = 1.6e-19
        c = 3e8
        wavelength = h / math.sqrt( 2 * m * charge * ht)
        relativistic_correction = 1 / math.sqrt(1 + ht * charge/(2 * m * c * c))
        return wavelength * relativistic_correction
