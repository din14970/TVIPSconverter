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
import re
from PyQt5.QtCore import QThread, pyqtSignal
from time import sleep

from enum import Enum

import logging
from .imagefun import (normalize_convert, bin2, gausfilter,
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


FILTER_DEFAULTS = {
               "useint": False, "whichint": 65536,
               "usebin": False, "whichbin": 1, "usegaus": False,
               "gausks": 8, "gaussig": 4, "usemed": False,
               "medks": 4, "usels": False, "lsmin": 10,
               "lsmax": 1000, "usecoffset": False
               }


VBF_DEFAULTS = {
                "calcvbf": True, "vbfrad": 10, "vbfxoffset": 0,
                "vbfyoffset": 0
                }


def _correct_column_offsets(image, thresholdmin=0, thresholdmax=30,
                            binning=1):
    """Do some kind of intensity correction, unsure reason"""
    pixperchannel = int(128 / binning)
    # binning has to be an integer
    if (128.0/binning != pixperchannel):
        logger.error("Can't figure out column offset dimension")
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
    subtracted = imtemp.astype(np.int64) - offsets[np.newaxis, :]
    subtracted[subtracted < 0] = 0
    subtracted.astype(image.dtype)
    return subtracted.reshape(image.shape)


def filter_image(imag, useint, whichint, usebin,
                 whichbin, usegaus, gausks,
                 gaussig, usemed, medks,
                 usels, lsmin, lsmax, usecoffset):
    """
    Filter an image and return the filtered image
    """
    # cut off too large intensities
    if useint:
        imag = np.where(imag > whichint, 0, imag)
    # binning by some factor
    if usebin:
        imag = bin2(imag, whichbin)
    # median filter
    if usemed:
        imag = medfilter(imag, medks)
    # apply gaussian filter
    if usegaus:
        imag = gausfilter(imag, gausks, gaussig)
    # linscale
    if usels:
        imag = normalize_convert(imag, lsmin, lsmax)
    # coffset
    if usecoffset:
        imag = _correct_column_offsets(imag)
    return imag


def getOriginalPreviewImage(path, improc, vbfsettings):
    rec = Recorder(path, improc=improc, vbfsettings=vbfsettings)
    firstframe = rec.read_first_frame()
    return firstframe


# def convertToHDF5(inpath, outpath, improc, vbfsettings, progbar=None):
#     rec = Recorder(inpath, improc=improc, vbfsettings=vbfsettings)
#     rec.convert_to_hdf5(outpath)
#     return True


class Recorder(QThread):

    increase_progress = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(self, path, improc=None, vbfsettings=None, numframes=None,
                 outputpath=None):
        QThread.__init__(self)
        logger.debug("Initializing recorder object")
        # filename
        self.filename = path  # first input file
        # general is the top header
        self.general = None
        self.dtype = None
        # to limit the number of frames read, really only for debugging
        self.numframes = numframes
        # keep a count of the number of files already read
        self.already_read = 0
        self.frameshape = None
        # image processing settings
        self.improc = improc
        if self.improc is None:
            self.improc = FILTER_DEFAULTS
        # vbf settings
        self.vbfproc = vbfsettings
        if self.vbfproc is None:
            self.vbfproc = VBF_DEFAULTS
        # output path
        self.outputpath = outputpath
        if self.outputpath is not None:
            self.outputpath = str(Path(self.outputpath))

    def run(self):
        # count = 0
        # while count < 100:
        #     count += 1
        #     self.docomplicatedthings(count)
        self.convert_HDF5()
        self.finish.emit()

    # def docomplicatedthings(self, count):
    #     sleep(0.2)
    #     self.increase_progress.emit(count)

    def convert_HDF5(self):
        """
        Convert to an HDF5 file
        """
        # for tracking progress
        self.total_size = self._get_total_scan_size()
        # progress as measured by files
        self.prog = 0
        # progress as measured within files
        self.fileprog = 0
        # HDF5 object to write the raw data to
        self.stream = h5py.File(self.outputpath, "w")
        self.streamgroup = self.stream.create_group("ImageStream")
        # also store immediately the processing info in attributes
        pg = self.streamgroup.create_group("Processing")
        for k, v in self.improc.items():
            pg.attrs[k] = v
        # This is important! also initializes the headers!
        firstframe = self.read_first_frame()
        pff = filter_image(firstframe, **self.improc)
        self.scangroup = self.stream.create_group("Scan")
        # do we need a virtual bright field calculated?
        if self.vbfproc["calcvbf"]:
            # make a VBF mask. For this we need a frame
            # ZOB center offset
            zoboffset = [self.vbfproc["vbfxoffset"],
                         self.vbfproc["vbfyoffset"]]
            radius = self.vbfproc["vbfrad"]
            # generate mask
            self.mask = self._virtual_bf_mask(pff, zoboffset, radius)
            sgp = self.scangroup.create_group("Processing")
            for k, v in self.vbfproc.items():
                sgp.attrs[k] = v
            self.vbfs = []
        # the start and stop frames
        # for convenience we also store rotation
        # indexes. For the end index we go backwards.
        self.start = None
        self.end = None
        self.rotidxs = []
        self._read_all_files()
        self._find_start_and_stop()
        self._save_preliminary_scan_info()
        self.stream.close()

    @staticmethod
    def valid_first_tvips_file(filename):
        """Check if the first tvips file is valid"""
        filpattern = re.compile(r".+\_([0-9]{3})\.(.*)")
        match = re.match(filpattern, filename)
        if match is not None:
            num, ext = match.groups()
            if ext != "tvips":
                raise ValueError(f"Invalid tvips file: extension {ext}, must "
                                 f"be tvips")
            if int(num) != 0:
                raise ValueError("Can only read video sequences starting with "
                                 "part 000")
            return True
        else:
            raise ValueError("Could not recognize as a valid tvips file")

    def read_first_frame(self):
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
            return frame
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
        # add the header to attributes
        for i in TVIPS_RECORDER_GENERAL_HEADER:
            self.streamgroup.attrs[i[0]] = self.general[i[0]]
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
            # progress
            self.prog += self.fileprog
            self.fileprog = 0

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
        # do calculations on the frame
        frame = filter_image(frame, **self.improc)
        # put the frame in the hdf5 file under the group
        c = f"{self.already_read}".zfill(6)
        ds = self.streamgroup.create_dataset(f"Frame_{c}", data=frame)
        for i in self.frame_header:
            ds.attrs[i[0]] = header[i[0]]
        # store the rotation index for finding start and stop later
        self.rotidxs.append(header['rotidx'])
        # immediately calculate and store the VBF intensity if required
        if self.vbfproc["calcvbf"]:
            vbf_int = frame[self.mask].mean()
            self.vbfs.append(vbf_int)
        # update the progressbar
        self.fileprog = fh.tell()
        self._update_gui_progess()

    def _update_gui_progess(self):
        """If using the GUI update features with progress"""
        value = int((self.prog+self.fileprog)/self.total_size*100)
        # logging.debug(f"We are at prog {self.prog}, fileprog {self.fileprog}"
        #               f"total size {self.total_size}: value={value}")
        self.increase_progress.emit(value)
        # self.progbar.setValue((self.prog+self.fileprog)//self.total_size*100)

    def _find_start_and_stop(self):
        # find out if it's the first or last frame
        previous = self.rotidxs[0]
        for j, i in enumerate(self.rotidxs):
            if i > previous:
                self.start = j-1
                logger.info(f"Found start at frame {j-1}")
                self.scangroup.attrs[
                    "start_frame"] = self.start
                break
            previous = i
        else:
            self.start = None
            self.scangroup.attrs[
                "start_frame"] = "None"
        # loop over it backwards to find the end
        # infact the index goes back to 1
        for j, i in reversed(list(enumerate(self.rotidxs))):
            if i > 1:
                self.end = j
                logger.info(f"Found final at frame {j}")
                self.scangroup.attrs[
                    "end_frame"] = self.end
                self.scangroup.attrs[
                    "final_rotinx"] = i
                self.final_rotinx = i
                break
        else:
            self.end = None
            self.scangroup.attrs[
                "end_frame"] = "None"
            self.final_rotinx = None
            self.scangroup.attrs[
                "final_rotinx"] = "None"
        # add a couple more attributes for good measure
        if self.end is not None and self.start is not None:
            self.scangroup.attrs[
                "ims_between_start_end"] = self.end-self.start
            self.scangroup.attrs[
                "total_scan_ims"] = len(self.rotidxs)

    def _save_preliminary_scan_info(self):
        # save rotation indexes and vbf intensities
        self.scangroup.create_dataset("rotation_indexes", data=self.rotidxs)
        if self.vbfproc["calcvbf"]:
            self.scangroup.create_dataset("vbf_intensities", data=self.vbfs)

    @staticmethod
    def _virtual_bf_mask(arr, centeroffsetpx=(0, 0), radiuspx=10):
        """Create virtual bright field mask"""
        xx, yy = np.meshgrid(np.arange(arr.shape[0], dtype=np.float),
                             np.arange(arr.shape[1], dtype=np.float))
        xx -= 0.5 * arr.shape[0] + centeroffsetpx[0]
        yy -= 0.5 * arr.shape[1] + centeroffsetpx[1]
        mask = np.hypot(xx, yy) < radiuspx
        return mask

    def determine_recorder_image_dimension(self, opts):
        # scan dimensions
        if (opts.dimension is not None):
            self.xdim, self.ydim = list(map(int, opts.dimension.split('x')))
        else:
            dim = math.sqrt(self.final_rotinx)
            if not dim == int(dim):
                raise ValueError("Can't determine correct image dimensions, "
                                 "please supply values manually (--dimension)")
            self.xdim, self.ydim = dim, dim
            logger.debug("Image dimensions: {}x{}".format(self.xdim,
                                                          self.ydim))

    def make_virtual_bf(self, opts):
        oimage = np.zeros((self.xdim*self.ydim), dtype=float)
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
