import numpy as np
import os.path
from tifffile import FileHandle
import math
import gc
import argparse
import tifffile
from types import SimpleNamespace
import h5py
from pathlib import Path

from enum import Enum

import logging
from .imagefun import (scale16to8, bin2, gausfilter,
                       medfilter)  # getElectronWavelength,

# Initialize the Logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

TVIPS_RECORDER_GENERAL_HEADER = [
    ('size', 'u4'),  # unused - likely the size of generalheader in bytes
    ('version', 'u4'),  # 1 or 2
    ('dimx', 'u4'),  # dp image size width
    ('dimy', 'u4'),  # dp image size height
    ('bitsperpixel', 'u4'),  # 8 or 16
    ('offsetx', 'u4'),  # generally 0
    ('offsety', 'u4'),
    ('binx', 'u4'),  # camera binning
    ('biny', 'u4'),
    ('pixelsize', 'u4'),  # nm, physical pixel size
    ('ht', 'u4'),  # high tension, voltage
    ('magtotal', 'u4'),  # magnification/camera length?
    ('frameheaderbytes', 'u4'),  # number of bytes per frame header
    ('dummy', 'S204'),  # just writes out TVIPS TVIPS TVIPS
    ]

TVIPS_RECORDER_FRAME_HEADER = [
    ('num', 'u4'),  # seems to cycle also
    ('timestamp', 'u4'),  # seconds since 1.1.1970
    ('ms', 'u4'),  # additional milliseconds to the timestamp
    ('LUTidx', 'u4'),  # always the same value
    ('fcurrent', 'f4'),  # 0 for all frames
    ('mag', 'u4'),  # same for all frames
    ('mode', 'u4'),  # 1 -> image 2 -> diff
    ('stagex', 'f4'),
    ('stagey', 'f4'),
    ('stagez', 'f4'),
    ('stagea', 'f4'),
    ('stageb', 'f4'),
    ('rotidx', 'u4'),
    ('temperature', 'f4'),  # cycles between 0.0 and 9.0 with step 1.0
    ('objective', 'f4'),  # kind of randomly between 0.0 and 1.0
    # for header version 2, some more data might be present
    ]


class Recorder(object):

    def __init__(self, opts):
        logger.debug("Initializing recorder object")
        self.filename = opts.input  # first input file
        assert os.path.exists(self.filename), "TVIPS file not found"
        assert self.filename.endswith(".tvips"), "File is not a tvips file"
        assert os.path.exists(opts.output), "Output folder doesn't exist"
        self.total_size = self._get_total_scan_size()
        # HDF5 object to write the raw data to
        oupstr = str(Path(opts.output+"/stream.hdf5"))
        self.stream = h5py.File(oupstr, "w")
        # general is the top header
        self.general = None
        self.dtype = None
        # to limit the number of frames read
        self.numframes = opts.numframes
        # keep a count of the number of files already read
        self.already_read = 0
        self.frameshape = None
        # basic image processing
        try:
            self.coffset = opts.coffset
        except Exception as e:
            logger.error(f"No parameters for coffset found, "
                         f"setting to None. Error: {e}")
            self.coffset = None
        # make a VBF mask. For this we need a frame
        self._read_first_frame()
        # ZOB center offset
        try:
            zoboffset = list(map(float, opts.vbfcenter.split('x')))
        except Exception as e:
            logger.error(f"No valid offsets defined for VBF spot, "
                         f"setting to zero. Error: {e}")
            zoboffset = list(map(float, (0, 0)))
        # generate mask
        try:
            radius = opts.vbfradius
        except Exception as e:
            logger.error(f"No valid radius defined for VBF spot, "
                         f"setting to 10. Error: {e}")
            radius = 10
        self.mask = self._virtual_bf_mask(zoboffset, radius)
        self.vbfs = []
        # the start and stop frames
        # for convenience we also store rotation
        # indexes. For the end index we go backwards.
        self.start = None
        self.end = None
        self.rotidxs = []
        self._read_all_files()
        self._find_start_and_stop()
        self.stream.close()

    def _read_first_frame(self):
        part = int(self.filename[-9:-6])
        if part != 0:
            raise ValueError("Can only read video sequences starting with "
                             "part 000")
        with open(self.filename, "rb") as f:
            fh = FileHandle(file=f)
            fh.seek(0)
            # skip general header from first file
            self._readGeneral(fh)
            # read first frame
            fh.read_record(self.frame_header)
            skip = self.inc - self.dt.itemsize
            fh.seek(skip, 1)
            # read frame
            frame = np.fromfile(
                        fh,
                        count=self.general.dimx*self.general.dimy,
                        dtype=self.dtype
                        )  # dtype=self.dtype,
            frame.shape = (self.general.dimx, self.general.dimy)
            self.firstframe = frame
        logger.debug("Read and stored the first frame")

    def _get_total_scan_size(self):
        """
        Get the total file size of the entire stream for tracking
        the progress
        """
        def get_filesize(fn):
            with open(fn, "rb") as f:
                fh = FileHandle(file=f)
                return fh.size
        sizes = self._scan_over_all_files(get_filesize)
        return sum(sizes)

    def _frames_exceeded(self):
        """Have we read the number of frames?"""
        if self.numframes is not None:
            if self.already_read >= self.numframes:
                logger.debug(f"{self.already_read} frames have been read. "
                             f"Quitting.")
                return True
        return False

    def _scan_over_all_files(self, func, *args, **kwargs):
        """Scan over all TVIPS file and perform a function on each.
        If the function returns something this returns the total."""
        results = []
        part = int(self.filename[-9:-6])
        if part != 0:
            raise ValueError("Can only read video sequences starting with "
                             "part 000")
        try:
            while True:
                fn = self.filename[:-9]+"{:03d}.tvips".format(part)
                if not os.path.exists(fn):
                    logger.debug(f"There is no file {fn}; breaking loop")
                    break
                logger.debug(f"Opening file {fn}")
                results.append(func(fn, *args, **kwargs))
                logger.debug(f"Finished reading file {fn}")
                part += 1
        except StopIteration:
            pass
        return results

    def _read_all_files(self):
        """Read in all data from all files and convert to hdf5"""
        # general info should already have been read by image 1
        # put header as a group into HDF5 group
        grp = self.stream.create_group("ImageStream")
        # add the header to attributes
        for i in TVIPS_RECORDER_GENERAL_HEADER:
            grp.attrs[i[0]] = self.general[i[0]]
        self._scan_over_all_files(self._readIndividualFile)

    def _readGeneral(self, fh):
        """Read the main header from the first file"""
        self.general = fh.read_record(TVIPS_RECORDER_GENERAL_HEADER)
        # changed np.uint16 to np.int16
        self.dtype = np.uint8 if self.general.bitsperpixel == 8 else np.uint16
        self.frameshape = (self.general.dimx, self.general.dimy)
        # set a few options for frame headers
        if self.general.version == 1:
            self.inc = 12
            self.frame_header = TVIPS_RECORDER_FRAME_HEADER
        elif self.general.version == 2:
            self.inc = self.general.frameheaderbytes
            self.frame_header = TVIPS_RECORDER_FRAME_HEADER
        else:
            raise NotImplementedError(f"Version {self.general.version} not "
                                      f"yet supported.")
        self.dt = np.dtype(self.frame_header)
        # make sure the record consumes less bytes than reported in the main
        # header
        assert self.inc >= self.dt.itemsize, ("The record consumes more bytes "
                                              "than stated in the main header")

    def _readIndividualFile(self, fn):
        """
        Extract the frames from the file and put into hdf5 file
        """
        logger.info("Reading {}".format(fn))
        part = int(fn[-9:-6])
        with open(fn, "rb") as f:
            fh = FileHandle(file=f)
            fh.seek(0)
            # read general header from first file
            if part == 0:
                self._readGeneral(fh)
            # read frames
            while fh.tell() < fh.size:
                self._readFrame(fh)
                self.already_read += 1  # increment number of frames
                if self._frames_exceeded():
                    raise StopIteration

    def _readFrame(self, fh, record=None):
        # read frame header
        header = fh.read_record(self.frame_header)
        logger.debug(f"{self.already_read}: Starting frame read "
                     f"(pos: {fh.tell()}). rot: {header['rotidx']}")
        skip = self.inc - self.dt.itemsize
        fh.seek(skip, 1)
        # read frame
        frame = np.fromfile(
                    fh,
                    count=self.general.dimx*self.general.dimy,
                    dtype=self.dtype
                    )  # dtype=self.dtype,
        frame.shape = (self.general.dimx, self.general.dimy)
        # do some basic calculations on the frame
        if (self.coffset is not None):
            frame = self._correct_column_offsets(frame)
        # put the frame in the hdf5 file under the group
        g = self.stream["ImageStream"]
        c = f"{self.already_read}".zfill(6)
        ds = g.create_dataset(f"Frame_{c}", data=frame)
        for i in self.frame_header:
            ds.attrs[i[0]] = header[i[0]]
        # store the rotation index for finding start and stop later
        self.rotidxs.append(header['rotidx'])
        # immediately calculate and store the VBF intensity
        vbf_int = frame[self.mask].mean()
        ds.attrs["VBF_intensity"] = vbf_int
        self.vbfs.append(vbf_int)

    def _find_start_and_stop(self):
        if self.rotidxs:
            # find out if it's the first or last frame
            previous = self.rotidxs[0]
            for j, i in enumerate(self.rotidxs):
                if i > previous:
                    self.start = j-1
                    logger.info(f"Found start at frame {j-1}")
                    self.stream["ImageStream"].attrs[
                        "start_frame"] = self.start
                    break
                previous = i
            # loop over it backwards to find the end
            previous = self.rotidxs[-1]
            for j, i in reversed(list(enumerate(self.rotidxs))):
                if i < previous:
                    self.end = j+1
                    logger.info(f"Found final at frame {j+1}")
                    self.stream["ImageStream"].attrs[
                        "start_frame"] = self.start
                    break
                previous = i

    def _virtual_bf_mask(self, centeroffsetpx=(0, 0), radiuspx=10):
        """Create virtual bright field mask"""
        arr = self.firstframe
        xx, yy = np.meshgrid(np.arange(arr.shape[0], dtype=np.float),
                             np.arange(arr.shape[1], dtype=np.float))
        xx -= 0.5 * arr.shape[0] + centeroffsetpx[0]
        yy -= 0.5 * arr.shape[1] + centeroffsetpx[1]
        mask = np.hypot(xx, yy) < radiuspx
        return mask

    def _correct_column_offsets(image, thresholdmin=0, thresholdmax=30,
                                binning=1):
        """Do some kind of intensity correction, unsure reason"""
        pixperchannel = int(128 / binning)
        # binning has to be an integer
        if (128.0/binning != pixperchannel):
            print("Can't figure out column offset dimension")
            return image
        numcol = int(image.shape[0] / 128 * binning)
        # fold up the image in a kind of 3D box
        imtemp = image.reshape((image.shape[0], pixperchannel, numcol))
        offsets = []
        for j in range(numcol):
            channel = imtemp[:, j, :]
            # pdb.set_trace()
            mask = np.bitwise_and(channel < thresholdmax,
                                  channel >= thresholdmin)
            value = np.mean(channel[mask])
            offsets.append(value)
        # apply offset correction to images
        offsets = np.array(offsets)
        return (imtemp - offsets[np.newaxis, :]).reshape(image.shape)

    def determine_recorder_image_dimension(self):
        # scan dimensions
        xdim, ydim = 0, 0
        if (opts.dimension is not None):
            xdim, ydim = list(map(int, opts.dimension.split('x')))
        else:
            dim = math.sqrt(len(rec.frames))
            if not dim == int(dim):
                raise ValueError("Can't determine correct image dimensions, "
                                 "please supply values manually (--dimension)")
            xdim, ydim = dim, dim
            logger.debug("Image dimensions: {}x{}".format(xdim, ydim))
        return xdim, ydim

    def make_virtual_bf(self):
        xdim, ydim = determine_recorder_image_dimension(self)
        oimage = np.zeros((xdim*ydim), dtype=rec.frames[0].dtype)
        for i, frame in enumerate(rec.frames[:xdim*ydim]):
            oimage[i] = frame[mask].mean()
        oimage.shape = (xdim, ydim)
        # correct for meander scanning of rotator
        if (opts.rotator):
            oimage[::2] = oimage[::2][:, ::-1]
            # correct for hysteresis
            if opts.hysteresis != 0:
                logger.info("Correcting for hysteresis...")
                oimage[::2] = np.roll(oimage[::2], opts.hysteresis, axis=1)
                logger.info("Rescaling to valid area...")
                oimage = oimage[:, opts.hysteresis:]
        logger.info("Writing out image")
        tifffile.imsave(opts.output, oimage)

    def _update_gui_progess(self):
        """If using the GUI update features with progress"""
        pass


class OutputTypes(Enum):
    IndividualTiff = "Individual"
    TiffStack = "TiffStack"
    TestFile = "TestFile"
    Blockfile = "blo"
    VirtualBF = "VirtualBF"

    def __str__(self):
        return self.value


def main(opts):
    """Main function that runs the transform depending on options"""
    logger.debug(str(opts))

    # read in file
    assert (os.path.exists(opts.input))

    # Definition of filter functions
    def dummy(x):
        return x

    # cut off too large intensities
    def cut_intensity(x):
        return np.where(x > opts.cutoff, 0, x)

    # median filter
    def med_filt(x):
        return medfilter(x, opts.median_kernel)

    # gaussian filter
    def gaus_filt(x):
        return gausfilter(x, opts.gaus_kernel, opts.gaus_sig)

    # binning by some factor
    def bnning(x):
        return bin2(x, opts.binning)

    # linscale
    def linscaling(x):
        min, max = map(float, opts.linscale.split('-'))
        return scale16to8(x, min, max)

    func1 = dummy
    if opts.cutoff is not None:
        func1 = cut_intensity

    func2 = dummy
    if opts.median_kernel is not None:
        func2 = med_filt

    func3 = dummy
    if opts.gaus_kernel is not None:
        func3 = gaus_filt

    func4 = dummy
    if opts.binning is not None:
        func4 = bnning

    func5 = dummy
    if opts.linscale is not None:
        func5 = linscaling

    # x is the frame
    def filterfunc(x):
        return func5(func2(func3(func4(func1(x)))))

    # read tvips file
    # default numframes is none
    rec = Recorder(opts)

    if (opts.depth is not None):
        dtype = np.dtype(opts.depth)  # parse dtype
        logger.debug("Mapping data to {}...".format(opts.depth))
        rec.frames = rec.frames.astype(dtype)
        logger.debug("Done Mapping")

    if (opts.otype == str(OutputTypes.IndividualTiff)):
        numframes = len(rec.frames)
        amount_of_digits = len(str(numframes-1))

        print("Start writing individual tif files to {}".format(opts.output))
        if not os.path.exists(opts.output):
            os.mkdir(opts.output)

        filename = "{}_{:0" + str(amount_of_digits) + "d}.tif"
        filename = os.path.join(opts.output, filename)

        for i, frame in enumerate(rec.frames):
            logger.info("Writing file {}".format(i))
            tifffile.imsave(filename.format(opts.fprefix, i), frame)
            logger.info("Finished writing file {}".format(i))

    elif (opts.otype == str(OutputTypes.TiffStack)):
        tifffile.imsave(opts.output, rec.toarray())

    elif (opts.otype == str(OutputTypes.TestFile)):
        tifffile.imsave(opts.output, rec.frames[0])
        logger.info("Wrote the test file")

    elif (opts.otype == str(OutputTypes.Blockfile) or
          opts.otype == OutputTypes.Blockfile):
        from . import blockfile
        xdim, ydim = determine_recorder_image_dimension()

        gc.collect()
        arr = rec.frames

        logger.debug("The frames are: {}".format(type(arr)))
        logger.debug("The x and ydim: {}x{}".format(xdim, ydim))
        if (len(arr) != xdim * ydim):
            # extend it to the requested dimensions
            missing = xdim*ydim - len(arr)
            arr = np.concatenate((arr, missing * [np.zeros_like(arr[0]), ]))
            logger.info("Data set filled up with {} frames for "
                        "matching requested dimensions".format(missing))
        arr.shape = (xdim, ydim, *arr[0].shape)
        # reorder meander
        arr[::2] = arr[::2][:, ::-1]
        # np.savez_compressed("c:\\temp\\dump.npz", data=arr)
        # TODO: check whether valid
        # apply hysteresis correction
        if opts.hysteresis != 0:
            logger.info("Correcting for hysteresis...")
            arr[::2] = np.roll(arr[::2], opts.hysteresis, axis=1)
            logger.info("Rescaling to valid area...")
            # arr = arr[:, opts.hysteresis:]
        # write out as tiffstack for now, later blo file with good header
        # tifffile.imsave(opts.output, arr, bigtiff=True)
        # calculate header flags
        # totalbinning = opts.binning * rec.general['binx']
        # wl = getElectronWavelength(1000.0 * rec.general['ht'])
        # pxsize = 1e-9 * rec.general['pixelsize']
        # wl in A * cl in cm * px per meter
        # ppcm = (wl*1e10 * rec.general['magtotal'] /
        #        (pxsize*totalbinning*opts.postmag))

        blockfile.file_writer_array(
                opts.output, arr, 5, 1.075,
                Camera_length=100.0*rec.general['magtotal'],
                Beam_energy=rec.general['ht']*1000,
                Distortion_N01=1.0, Distortion_N09=1.0,
                Note="Cheers from TVIPS!")
        logger.debug("Finished writing the blo file")
    else:
        raise ValueError("No output type specified (--otype)")


def mainCLI():
    """Main function as run from the command line

    The argparse arguments are passed directly to the main function
    """
    parser = argparse.ArgumentParser(
     description='Process .tvips recorder format')

    parser.add_argument('--otype', type=OutputTypes, choices=list(OutputTypes),
                        help='Output format')
    parser.add_argument("--numframes", type=int, default=None,
                        help="Limit data to the first n frames")
    parser.add_argument("--binning", type=int, default=None, help="Bin data")
    parser.add_argument("--dumpheaders", action="store_true", default=False,
                        help="Dump headers")
    parser.add_argument('--depth', choices=("uint8", "uint16", "float"),
                        default=None)
    parser.add_argument('--linscale',
                        help=("Scale 16 bit data linear to 8 bit using the "
                              "given range. Eg. 100-1000. Default: min, max"),
                        default=None)
    parser.add_argument('--coffset', action='store_true', default=False)
    # Virtual BF/blo options
    parser.add_argument('--vbfcenter', default='0.0x0.0',
                        help='Offset to center of Zero Order Beam')
    parser.add_argument('--vbfradius', default=10.0, type=float,
                        help='Integration disk radius')
    parser.add_argument('--dimension',
                        help='Output dimensions, default: sqrt(#images)^2')
    parser.add_argument('--rotator', action="store_true",
                        help="Pick only valid rotator frames")
    parser.add_argument('--hysteresis', default=0, type=int,
                        help='Move every second row by n pixel')
    parser.add_argument('--postmag', default=1.0, type=float,
                        help="Apply a mag correction")
    parser.add_argument('--skip', default=0, type=int,
                        help='Skip # images at the beginning')
    parser.add_argument('--truncate', default=0, type=int,
                        help='Truncate # images at the end')
    # self added
    parser.add_argument('--fprefix', default=None,
                        help='Output Image/file name')
    parser.add_argument('--cutoff', default=None, type=int,
                        help='intensity cut off set to 0')
    parser.add_argument('--median_kernel', default=None, type=int,
                        help='median kernel size')
    parser.add_argument('--gaus_kernel', default=None, type=int,
                        help='gausian kernel size')
    parser.add_argument('--gaus_sig', default=None, type=int,
                        help='gaus filter sigma')
    parser.add_argument('input', help="Input filename, must be _000.tvips")
    parser.add_argument('output', help="Output dir or output filename")
    opts = parser.parse_args()
    main(opts)


def mainUI(**k):
    """Main function as run from the gui

    The arguments k represent those read from the gui. These are converted
    to arguments that the main converter function understands.
    """
    ks = SimpleNamespace(**k)
    d = {}
    if ks.oupt == ".blo":
        d["otype"] = "blo"
        d["output"] = ks.oup+"/{}.blo".format(ks.pref)
    if ks.oupt == "list of .tiff":
        d["otype"] = "Individual"
        d["output"] = ks.oup
        d["fprefix"] = ks.pref
    d["numframes"] = None
    if ks.use_bin == 2:
        d["binning"] = ks.bin_fac
    else:
        d["binning"] = None
    d["depth"] = ks.dep
    if ks.use_scaling == 2:
        d["linscale"] = "{}-{}".format(ks.scalemin, ks.scalemax)
    else:
        d["linscale"] = None
    d["dimension"] = "{}x{}".format(ks.sdx, ks.sdy)
    d["rotator"] = (ks.use_rotator == 2)
    d["dumpheaders"] = False
    d["coffset"] = False
    d["postmag"] = 1.0
    if ks.use_hyst == 2:
        d["hysteresis"] = ks.hyst_val
    else:
        d["hysteresis"] = 0
    d["skip"] = ks.skip
    d["truncate"] = ks.trunc
    d["input"] = ks.inp
    # self created args
    if ks.useintcut == 2:
        d["cutoff"] = ks.intcut
    else:
        d["cutoff"] = None
    if ks.use_med == 2:
        d["median_kernel"] = ks.med_ks
    else:
        d["median_kernel"] = None
    if ks.use_gauss == 2:
        d["gaus_kernel"] = ks.gauss_ks
        d["gaus_sig"] = ks.gauss_sig
    else:
        d["gaus_kernel"] = None
        d["gaus_sig"] = None
    opts = SimpleNamespace(**d)
    main(opts)


def createOneImageUI(**k):
    """
    Extract the first image from the file using the main function

    Some arguments are read from the GUI and sent into a simplenamespace.
    Arguments mainly relate to processing of the image.
    """
    # extracted info from the gui
    ks = SimpleNamespace(**k)
    # create the args sent to main
    d = {}
    d["otype"] = "TestFile"
    d["numframes"] = 1
    if ks.use_bin == 2:
        d["binning"] = ks.bin_fac
    else:
        d["binning"] = None
    d["depth"] = ks.dep
    if ks.use_scaling == 2:
        d["linscale"] = "{}-{}".format(ks.scalemin, ks.scalemax)
    else:
        d["linscale"] = None
    d["dimension"] = "{}x{}".format(ks.sdx, ks.sdy)
    d["rotator"] = False  # (ks.use_rotator==2)
    d["dumpheaders"] = False
    d["coffset"] = False
    d["hysteresis"] = 0
    d["skip"] = 0  # ks.skip
    d["truncate"] = 0  # ks.trunc
    d["input"] = ks.inp
    d["output"] = "./temp.tiff"
    # self created args
    if ks.useintcut == 2:
        d["cutoff"] = ks.intcut
    else:
        d["cutoff"] = None
    if ks.use_med == 2:
        d["median_kernel"] = ks.med_ks
    else:
        d["median_kernel"] = None
    if ks.use_gauss == 2:
        d["gaus_kernel"] = ks.gauss_ks
        d["gaus_sig"] = ks.gauss_sig
    else:
        d["gaus_kernel"] = None
        d["gaus_sig"] = None
    opts = SimpleNamespace(**d)
    main(opts)


if __name__ == "__main__":
    mainCLI()
