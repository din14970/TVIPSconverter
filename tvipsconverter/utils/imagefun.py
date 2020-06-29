import numpy as np
import math
from scipy.ndimage import median_filter, convolve
from PIL import Image


def _get_dtype_min_max(dtype):
    """
    Return (min, max) of a numpy integer or float dtype

    For floats it just returns 0 and 1
    """
    if dtype == np.float or dtype == np.float32 or dtype == np.float64:
        max = 1  # np.finfo(dtype).max
        min = 0  # np.finfo(dtype).min
    elif (dtype == np.int8 or dtype == np.uint8 or
          dtype == np.int16 or dtype == np.uint16 or
          dtype == np.int32 or dtype == np.uint32 or
          dtype == np.int64 or dtype == np.uint64):
        max = np.iinfo(dtype).max
        min = np.iinfo(dtype).min
    else:
        raise ValueError("Unrecognized or unsupported type: {dtype}")
    return (min, max)


def normalize_convert(img, min=None, max=None, dtype=None):
    """
    Linscales an array and converts dtype

    Only unsigned integer dtypes are accepted

    Parameters
    ----------
    img : array
        The 2D image
    dtype : numpy.dtype, optional
        output type (options: numpy int and float types).
        Defaults to the same datatype as the image.
        If any kind of float is chosen
        the image is renormalized to between 0 and 1.
    min : int, optional
        the value in the img to map to the minimum. Everything
        below is set to minimum. Defaults to the minimum value
        in the array.
    max : int, optional
        the value in the img to map to the maximum. Everything
        above is set to maximum. Defaults to the maximum value
        in the array.

    Returns
    -------
    img_new : array
        The rescaled and retyped image
    """
    if dtype is None:
        dtype = img.dtype
    nmin, nmax = _get_dtype_min_max(dtype)
    return linscale(img, min, max, nmin, nmax, dtype)


def linscale(arr, min=None, max=None, nmin=0, nmax=1, dtype=np.float):
    """
    Rescale image intensities

    Rescale the image intensities to a new scale. The value
    min and everything below gets mapped to nmin, max and everything
    above gets mapped to nmax. By default the minimum and maximum
    get mapped to 0 and 1.

    Parameters
    ----------
    arr : array-like object
        The image to normalize
    min : float, optional
        The intensity to map to the new minimum value. Defaults to
        the minimum of the provided array.
    max : float, optional
        The intensity to map to the new maximum value. Defaults to
        the maximum of the provided array.
    nmin : float, optional
        The new minimum of the image. Defaults to 0.
    nmax : float, optional
        The new maximum of the image. Defaults to 1. For 8-bit images use 255.
        For 16-bit images use 65535.
    dtype : type, optional
        The data type of the output array. Defaults to float. See the possible
        data types here:
        <https://docs.scipy.org/doc/numpy/user/basics.types.html>

    Returns
    -------
    result : array
        Intensity-rescaled image

    Notes
    -----
    The type recasting happens in an 'unsafe' manner. That is, if elements
    have float values like 0.99, recasting to np.uint8 will turn this into 0
    and not 1.

    Examples
    --------
    >>>s=np.array([[1, 2], [3, 4]])
    >>>linscale(s)
    array([[0.        , 0.33333333],
           [0.66666667, 1.        ]])
    >>>linscale(s, min=2, max=3, nmin=1, nmax=2, dtype=np.uint8)
    array([[1, 1],
           [2, 2]], dtype=uint8)
    """
    workarr = arr.copy()
    if min is None:
        min = workarr.min()
    else:
        workarr[workarr < min] = min
    if max is None:
        max = workarr.max()
    else:
        workarr[workarr > max] = max

    a = (nmax-nmin)/(max-min)
    result = (workarr-min)*a+nmin
    return result.astype(dtype)


# def scale16to8(arr, min=None, max=None):
#     if min is None:
#         min = arr.min()
#     else:
#         arr[arr<min] = min
#     if max is None:
#         max = arr.max()
#     else:
#         arr[arr>max] = max
#
#     return ((arr-min)*255//(max-min))


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gausfilter(arr, ks, sig):
    kernel = matlab_style_gauss2D(shape=(ks, ks), sigma=sig)
    return convolve(arr, kernel, mode="constant", cval=0)


def medfilter(arr, ks):
    return median_filter(arr, size=ks, mode="constant", cval=0)


def findoutliers(raw, percent=0.07):
    """
    findoutliers (array, percent=0.07)

    Uses returns min and max values of the array with the upper and
    lower percent pixel values suppressed.
    """
    cnts, edges = np.histogram(raw, bins=2**16)
    stats = np.zeros((2, 2**16), dtype=np.int)
    stats[0] = np.cumsum(cnts)  # low
    stats[1] = np.cumsum(cnts[::-1])  # high
    thresh = stats > percent * raw.shape[0] * raw.shape[1]
    min = (np.where(thresh[0]))[0][0]
    max = 2**16 - (np.where(thresh[1]))[0][0]
    return edges[min], edges[max+1]


def suppressoutliers(raw, percent=0.07):
    min, max = findoutliers(raw, percent)

    raw[raw < min] = min
    raw[raw > max] = max
    return raw


def bin2(a, factor):
    assert len(a.shape) == 2
    imag = Image.fromarray(a)
    # binned = Image.resize(imag, (a.shape[0]//factor, a.shape[1]//factor),
    #                       resample=Image.BILINEAR)
    binned = np.array(imag.resize((a.shape[0]//factor, a.shape[1]//factor),
                      resample=Image.NEAREST))
    print(binned.dtype)
    return binned


def getElectronWavelength(ht):
    # ht in Volts, length unit in meters
    h = 6.6e-34
    m = 9.1e-31
    charge = 1.6e-19
    c = 3e8
    wavelength = h / math.sqrt(2 * m * charge * ht)
    relativistic_correction = 1 / math.sqrt(1 + ht * charge/(2 * m * c * c))
    return wavelength * relativistic_correction
