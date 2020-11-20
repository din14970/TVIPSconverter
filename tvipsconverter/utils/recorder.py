﻿import numpy as np
import os.path
from scipy.ndimage import gaussian_filter
from tifffile import FileHandle
import math
import h5py
from pathlib import Path
import re
from PyQt5.QtCore import QThread, pyqtSignal

import logging
from .imagefun import normalize_convert, bin2, bin_box, gausfilter, medfilter

# Initialize the Logger
logger = logging.getLogger(__name__)

TVIPS_RECORDER_GENERAL_HEADER = [
    ("size", "u4"),  # unused - likely the size of generalheader in bytes
    ("version", "u4"),  # 1 or 2
    ("dimx", "u4"),  # dp image size width
    ("dimy", "u4"),  # dp image size height
    ("bitsperpixel", "u4"),  # 8 or 16
    ("offsetx", "u4"),  # generally 0
    ("offsety", "u4"),
    ("binx", "u4"),  # camera binning
    ("biny", "u4"),
    ("pixelsize", "u4"),  # nm, physical pixel size
    ("ht", "u4"),  # high tension, voltage
    ("magtotal", "u4"),  # magnification/camera length?
    ("frameheaderbytes", "u4"),  # number of bytes per frame header
    ("dummy", "S204"),  # just writes out TVIPS TVIPS TVIPS
]

TVIPS_RECORDER_FRAME_HEADER = [
    ("num", "u4"),  # seems to cycle also
    ("timestamp", "u4"),  # seconds since 1.1.1970
    ("ms", "u4"),  # additional milliseconds to the timestamp
    ("LUTidx", "u4"),  # always the same value
    ("fcurrent", "f4"),  # 0 for all frames
    ("mag", "u4"),  # same for all frames
    ("mode", "u4"),  # 1 -> image 2 -> diff
    ("stagex", "f4"),
    ("stagey", "f4"),
    ("stagez", "f4"),
    ("stagea", "f4"),
    ("stageb", "f4"),
    ("rotidx", "u4"),
    ("temperature", "f4"),  # cycles between 0.0 and 9.0 with step 1.0
    ("objective", "f4"),  # kind of randomly between 0.0 and 1.0
    # for header version 2, some more data might be present
]


FILTER_DEFAULTS = {
    "useint": False,
    "whichint": 65536,
    "usebin": False,
    "whichbin": 1,
    "usegaus": False,
    "gausks": 8,
    "gaussig": 4,
    "usemed": False,
    "medks": 4,
    "usels": False,
    "lsmin": 10,
    "lsmax": 1000,
    "usecoffset": False,
}


VBF_DEFAULTS = {"calcvbf": True, "vbfrad": 10, "vbfxoffset": 0, "vbfyoffset": 0}


def _correct_column_offsets(image, thresholdmin=0, thresholdmax=30, binning=1):
    """Do some kind of intensity correction, unsure reason"""
    pixperchannel = int(128 / binning)
    # binning has to be an integer
    if 128.0 / binning != pixperchannel:
        logger.error("Can't figure out column offset dimension")
        return image
    numcol = int(image.shape[0] / 128 * binning)
    # fold up the image in a kind of 3D box
    imtemp = image.reshape((image.shape[0], pixperchannel, numcol))
    offsets = []
    for j in range(numcol):
        channel = imtemp[:, j, :]
        # pdb.set_trace()
        mask = np.bitwise_and(channel < thresholdmax, channel >= thresholdmin)
        value = np.mean(channel[mask])
        offsets.append(value)
    # apply offset correction to images
    offsets = np.array(offsets)
    subtracted = imtemp.astype(np.int64) - offsets[np.newaxis, :]
    subtracted[subtracted < 0] = 0
    subtracted.astype(image.dtype)
    return subtracted.reshape(image.shape)


def filter_image(
    imag,
    useint,
    whichint,
    usebin,
    whichbin,
    usegaus,
    gausks,
    gaussig,
    usemed,
    medks,
    usels,
    lsmin,
    lsmax,
    usecoffset,
    bintype,
):
    """
    Filter an image and return the filtered image
    """
    # cut off too large intensities
    if useint:
        imag = np.where(imag > whichint, 0, imag)
    # binning by some factor
    if usebin:
        if bintype:  # if True use interpolate, else box
            imag = bin2(imag, whichbin)
        else:
            # test binning factor will work
            if all(not i % whichbin for i in imag.shape):
                imag = bin_box(imag, whichbin)
            else:
                logger.warning(
                    "array shape is not factorisable by factor, using decimation instead."
                )
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


def getOriginalPreviewImage(path, improc, vbfsettings, frame=0):
    rec = Recorder(path, improc=improc, vbfsettings=vbfsettings)
    firstframe = rec.read_frame(frame)
    return firstframe


def write_scan_parameters_hdf5(path, **parameters):
    """
    Write parameters as attrs to a hdf5 file.

    Parameters are written under h5['Scan'].attrs
    """
    with h5py.File(path, "r+") as f:
        for key, val in parameters.items():
            f["Scan"].attrs[key] = val


class Recorder(QThread):

    increase_progress = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(
        self,
        path,
        improc=None,
        vbfsettings=None,
        outputpath=None,
        imrange=(None, None),
        **options,
    ):
        QThread.__init__(self)
        logger.debug("Initializing recorder object")
        # filename
        self.filename = path  # first input file
        with open(self.filename, "rb") as f:
            fh = FileHandle(file=f)
            fh.seek(0)
            # defines: self.general, self.dtype, self.frameshape
            # self.inc, self.frame_header
            self._readGeneral(fh)
            self.generalheadersize = fh.tell()
        # keep a count of the number of files already read
        self.current_frame = 0
        # a map of the file, which bytes start where
        self.ranges = self._get_files_ranges()
        # image range to consider
        self.startim, self.finalim = imrange
        self.startbyte = None
        self.endbyte = None
        # for tracking progress
        self.total_size = self._get_total_scan_size()
        if self.startim is not None:
            self.startbyte = self._get_frame_byte_position(self.startim)
            if self.startbyte > self.total_size:
                raise ValueError("Start frame out of bounds")
        else:
            self.startbyte = 0
            self.startim = 0
        if self.finalim is not None:
            self.endbyte = self._get_frame_byte_position(self.finalim)
            if self.endbyte > self.total_size:
                raise ValueError("End frame out of bounds")
        else:
            self.endbyte = self.total_size
            self.finalim = self._get_byte_frame_position(self.endbyte)
        print(f"We expect the last frame to be {self.startim}")
        print(f"We expect the last frame to be {self.finalim}")
        if self.startbyte >= self.endbyte:
            raise ValueError("Start frame must be less than end frame")
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

        # other options, added 2.11.2020
        self.options = options
        # print(options, self.options)

    def run(self):
        self.convert_HDF5()
        self.finish.emit()

    def convert_HDF5(self):
        """
        Convert to an HDF5 file
        """
        # progress as measured by files
        self.prog = 0
        # progress as measured within files
        self.cur_pos_in_file = 0
        # HDF5 object to write the raw data to
        try:
            self.stream = h5py.File(self.outputpath, "w")
        except Exception:
            raise Exception("This file already seems open somewhere else")
        self.streamgroup = self.stream.create_group("ImageStream")
        # also store immediately the processing info in attributes
        pg = self.streamgroup.create_group("Processing")
        for k, v in self.improc.items():
            pg.attrs[k] = v
        # This is important! also initializes the headers!
        firstframe = self.read_frame(0)
        pff = filter_image(firstframe, **self.improc)

        # initialise maximum_image if checkBox checked
        # print(f"average image: {'calcmax' in self.options} {self.options['calcmax']}")
        if "calcmax" in self.options and self.options["calcmax"]:
            # print("creating maxiumum image")
            self.maximum_image = np.zeros_like(pff)

        # print(f"average image: {'calcave' in self.options} {self.options['calcave']}")
        # initialise average_image if checkBox checked
        if "calcave" in self.options and self.options["calcave"]:
            # print("creating average_image")
            # as datatype is typcially 8- or 16-bit then 32-bit image should be fine
            self.average_image = np.zeros_like(pff, dtype=np.float32)

        self.scangroup = self.stream.create_group("Scan")
        # do we need a virtual bright field calculated?
        if self.vbfproc["calcvbf"]:
            # make a VBF mask. For this we need a frame
            # ZOB center offset
            zoboffset = [self.vbfproc["vbfxoffset"], self.vbfproc["vbfyoffset"]]
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
        # for calculating where the scan began and stopped
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
                raise ValueError(
                    f"Invalid tvips file: extension {ext}, must " f"be tvips"
                )
            if int(num) != 0:
                raise ValueError(
                    "Can only read video sequences starting with " "part 000"
                )
            return True
        else:
            raise ValueError("Could not recognize as a valid tvips file")

    def read_frame(self, frame=0):
        bite_start = self._get_frame_byte_position(frame)
        toopen = self._get_byte_filename(bite_start)
        with open(toopen, "rb") as f:
            fh = FileHandle(file=f)
            fh.seek(bite_start - self.ranges[toopen][0])
            frame = np.fromfile(
                fh, count=self.general.dimx * self.general.dimy, dtype=self.dtype
            )
            frame.shape = (self.general.dimx, self.general.dimy)
            return frame

    def _get_frame_filename(self, frame):
        """Get the filename where a certain frame is in"""
        bite_start = self._get_frame_byte_position(frame)
        return self._get_byte_filename(bite_start)

    def _get_byte_filename(self, b):
        """Get the filename where the byte b is in"""
        # find in which file I must search
        toopen = ""
        for i in self.ranges:
            mi, ma = self.ranges[i]
            if b < ma and b > mi:
                toopen = i
                break
        else:
            raise ValueError(f"Byte {b} is out of bounds")
        return toopen

    def _get_frame_byte_position(self, frame):
        """Get the byte where a frame starts"""
        frame_byte_size = (
            self.general.dimx * self.general.dimy * self.general.bitsperpixel // 8
        )
        bite_start = (
            self.generalheadersize
            + self.general.frameheaderbytes
            + (frame_byte_size + self.general.frameheaderbytes) * frame
        )
        return bite_start

    def _get_frame_byte_position_in_file(self, frame):
        """Get the byte where a frame starts in a file"""
        abs_pos = self._get_frame_byte_position(frame)
        fn = self._get_byte_filename(abs_pos)
        mi, _ = self.ranges[fn]
        return abs_pos - mi

    def _get_byte_frame_position(self, byte_start):
        """Get the frame index closest corresponding to a byte"""
        frame_byte_size = (
            self.general.dimx * self.general.dimy * self.general.bitsperpixel // 8
        )
        frame = (
            byte_start - self.generalheadersize - self.general.frameheaderbytes
        ) // (frame_byte_size + self.general.frameheaderbytes)
        return frame

    def _get_files_size_dictionary(self):
        """
        Get a dictionary of files (keys) and total size (values)
        """

        def get_filesize(fn):
            with open(fn, "rb") as f:
                fh = FileHandle(file=f)
                return fn, fh.size

        sizes = self._scan_over_all_files(get_filesize)
        return dict(sizes)

    def _get_files_ranges(self):
        """
        Get a dictionary of files (keys) and the start and end
        """
        sizes = self._get_files_size_dictionary()
        ranges = {}
        starts = 0
        for i in sizes:
            ends = starts + sizes[i]
            ranges[i] = (starts, ends)
            starts = ends
        return ranges

    def _get_total_scan_size(self):
        """
        Get the total file size of the entire stream for tracking
        the progress
        """
        sizes = self._get_files_size_dictionary()
        size = sum(sizes.values())
        return size

    def _get_total_scan_size_limited(self):
        """Scan size considering start and end frame"""
        if self.endbyte:
            size = self.endbyte
        if self.startbyte:
            size = size - self.startbyte

    def _frames_exceeded(self):
        """Have we read the number of frames?"""
        if self.finalim is not None:
            if self.current_frame >= self.finalim:
                logger.debug(f"We are at frame {self.current_frame}. " f"Quitting.")
                return True
        return False

    def _scan_over_all_files(self, func, *args, **kwargs):
        """Scan over all TVIPS file and perform a function on each.
        If the function returns something this returns the total."""
        results = []
        part = int(self.filename[-9:-6])
        if part != 0:
            raise ValueError("Can only read video sequences starting with " "part 000")
        try:
            while True:
                fn = self.filename[:-9] + "{:03d}.tvips".format(part)
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
            raise NotImplementedError(
                f"Version {self.general.version} not " f"yet supported."
            )
        self.dt = np.dtype(self.frame_header)
        # make sure the record consumes less bytes than reported in the main
        # header
        assert self.inc >= self.dt.itemsize, (
            "The record consumes more bytes " "than stated in the main header"
        )

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
                if self.startim > self.current_frame:
                    self.current_frame += 1
                    current_byte = self._get_frame_byte_position_in_file(
                        self.current_frame
                    )
                    fh.seek(current_byte - self.general.frameheaderbytes)
                    continue
                if self.finalim < self.current_frame:
                    break
                self._readFrame(fh)
                self.current_frame += 1  # increment number of frames
                # update the progressbar
                self._update_gui_progess()

    def _readFrame(self, fh, record=None):
        # read frame header
        header = fh.read_record(self.frame_header)
        logger.debug(
            f"{self.current_frame}: Starting frame read "
            f"(pos: {fh.tell()}). rot: {header['rotidx']}"
        )
        skip = self.inc - self.dt.itemsize
        fh.seek(skip, 1)
        # read frame
        frame = np.fromfile(
            fh, count=self.general.dimx * self.general.dimy, dtype=self.dtype
        )
        frame.shape = (self.general.dimx, self.general.dimy)
        # do calculations on the frame
        frame = filter_image(frame, **self.improc)
        # put the frame in the hdf5 file under the group
        c = f"{self.current_frame-self.startim}".zfill(6)
        ds = self.streamgroup.create_dataset(f"Frame_{c}", data=frame)
        for i in self.frame_header:
            ds.attrs[i[0]] = header[i[0]]
        # store the rotation index for finding start and stop later
        self.rotidxs.append(header["rotidx"])
        # immediately calculate and store the VBF intensity if required
        if self.vbfproc["calcvbf"]:
            vbf_int = frame[self.mask].mean()
            self.vbfs.append(vbf_int)

        # calculate images as specified in the options
        if "calcmax" in self.options and self.options["calcmax"]:
            # maximum_image should already be initialised in self.convert_HDF5
            self.maximum_image = np.stack((self.maximum_image, frame), axis=0).max(
                axis=0
            )

        if "calcave" in self.options and self.options["calcave"]:
            # average_image should already be initialised in self.convert_HDF5
            self.average_image = np.stack(
                (
                    self.average_image,
                    # frame is scaled as a function of 1/total_frames and then summed (average)
                    (1 / (self.finalim - self.startim))
                    * frame.astype(self.average_image.dtype),
                ),
                axis=0,
            ).sum(axis=0)

        if (
            "refine_center" in self.options and self.options["refine_center"][0]
        ):  # make sure is checked
            side, sigma = self.options["refine_center"][1:]
            center = np.array(frame.shape) // 2

            crop = frame[
                center[0] - side // 2 : center[0] + side // 2,
                center[1] - side // 2 : center[1] + side // 2,
            ]
            # blur crop and find maximum -> use as center location
            blurred = gaussian_filter(crop, sigma, mode="nearest")
            # add crop offset (center - side//2) to get actual location on frame
            ds.attrs["Center location"] = np.unravel_index(
                blurred.argmax(), crop.shape
            ) + (center - side // 2)

    def _update_gui_progess(self):
        """If using the GUI update features with progress"""
        value = int(
            (self.current_frame - self.startim) / (self.finalim - self.startim) * 100
        )
        self.increase_progress.emit(value)

    def _find_start_and_stop(self):
        """ find out if it's the first or last frame """
        previous = self.rotidxs[0]
        for j, i in enumerate(self.rotidxs):
            if i > previous:
                self.start = j - 1
                logger.info(f"Found start at frame {j-1}")
                self.scangroup.attrs["start_frame"] = self.start
                break
            previous = i
        else:
            self.start = None
            self.scangroup.attrs["start_frame"] = "None"
        # loop over it backwards to find the end
        # infact the index goes back to 1
        for j, i in reversed(list(enumerate(self.rotidxs))):
            if i > 1:
                self.end = j
                logger.info(f"Found final at frame {j}")
                self.scangroup.attrs["end_frame"] = self.end
                self.scangroup.attrs["final_rotinx"] = i
                self.final_rotinx = i
                break
        else:
            self.end = None
            self.scangroup.attrs["end_frame"] = "None"
            self.final_rotinx = None
            self.scangroup.attrs["final_rotinx"] = "None"
        # add a couple more attributes for good measure
        self.scangroup.attrs["total_stream_frames"] = len(self.rotidxs)
        if self.end is not None and self.start is not None:
            self.scangroup.attrs["ims_between_start_end"] = self.end - self.start

    def _save_preliminary_scan_info(self):
        # save rotation indexes and vbf intensities
        self.scangroup.create_dataset("rotation_indexes", data=self.rotidxs)
        if self.vbfproc["calcvbf"]:
            self.scangroup.create_dataset("vbf_intensities", data=self.vbfs)

        # save options
        if "calcmax" in self.options and self.options["calcmax"]:
            self.scangroup.create_dataset("maximum_image", data=self.maximum_image)
        if "calcave" in self.options and self.options["calcave"]:
            self.scangroup.create_dataset("average_image", data=self.average_image)

    @staticmethod
    def _virtual_bf_mask(arr, centeroffsetpx=(0, 0), radiuspx=10):
        """Create virtual bright field mask"""
        xx, yy = np.meshgrid(
            np.arange(arr.shape[0], dtype=np.float),
            np.arange(arr.shape[1], dtype=np.float),
        )
        xx -= 0.5 * arr.shape[0] + centeroffsetpx[0]
        yy -= 0.5 * arr.shape[1] + centeroffsetpx[1]
        mask = np.hypot(xx, yy) < radiuspx
        return mask

    def determine_recorder_image_dimension(self, opts):
        # scan dimensions
        if opts.dimension is not None:
            self.xdim, self.ydim = list(map(int, opts.dimension.split("x")))
        else:
            dim = math.sqrt(self.final_rotinx)
            if not dim == int(dim):
                raise ValueError(
                    "Can't determine correct image dimensions, "
                    "please supply values manually (--dimension)"
                )
            self.xdim, self.ydim = dim, dim
            logger.debug("Image dimensions: {}x{}".format(self.xdim, self.ydim))


class hdf5Intermediate(h5py.File):
    """This class represents the intermediate hdf5 file handle"""

    def __init__(self, filepath, mode="r"):
        super().__init__(filepath, mode)
        (
            self.total_frames,
            self.start_frame,
            self.end_frame,
            self.final_rotator,
            dim,
            self.imdimx,
            self.imdimy,
        ) = self.get_scan_info()
        self.sdimx = dim
        self.sdimy = dim

    def get_scan_info(self):
        """Return total frames, start frame, end frame, guessed scan dimensions
        as well as image dimensions"""
        try:
            tot = self["Scan"].attrs["total_stream_frames"]
            tot = int(tot)
        except Exception as e:
            logger.debug(f"Total stream frames not found, error: {e}")
            tot = None
        try:
            start = self["Scan"].attrs["start_frame"]
            start = int(start)
        except Exception as e:
            logger.debug(f"start frame not found, error: {e}")
            start = None
        try:
            end = self["Scan"].attrs["end_frame"]
            end = int(end)
        except Exception as e:
            logger.debug(f"end frame not found, error: {e}")
            end = None
        try:
            finrot = self["Scan"].attrs["final_rotinx"]
            finrot = int(finrot)
        except Exception as e:
            logger.debug(f"final rotator index not found, error: {e}")
            finrot = None
        try:
            if (
                "scan_dim_x" in self["Scan"].attrs
                and "scan_dim_y" in self["Scan"].attrs
            ):
                dim = (
                    self["Scan"].attrs["scan_dim_x"],
                    self["Scan"].attrs["scan_dim_y"],
                )

            else:
                dim = round(np.sqrt(finrot), 6)
                if not dim == int(dim):
                    raise Exception
                else:
                    dim = int(dim)
        except Exception:
            logger.debug("Could not calculate scan dimensions")
            dim = None
        try:
            imdimx, imdimy = self["ImageStream"]["Frame_000000"].shape
        except Exception:
            imdimx = None
            imdimy = None
        return (tot, start, end, finrot, dim, imdimx, imdimy)

    def get_vbf_image(
        self,
        sdimx=None,
        sdimy=None,
        start_frame=None,
        end_frame=None,
        hyst=0,
        snakescan=True,
    ):
        # try to get the rotator data
        try:
            vbfs = self["Scan"]["vbf_intensities"][:]
        except Exception:
            raise Exception(
                "No VBF information found in dataset, please "
                "calculate from TVIPS file."
            )
        logger.debug("Succesfully imported vbf intensities")
        logger.debug("Now calculating scan indexes")
        scan_indexes = self.calculate_scan_export_indexes(
            sdimx=sdimx,
            sdimy=sdimy,
            start_frame=start_frame,
            end_frame=end_frame,
            hyst=hyst,
            snakescan=snakescan,
        )
        logger.debug("Calculated scan indexes")
        if sdimx is None:
            sdimx = self.sdimx
        if sdimy is None:
            sdimy = self.sdimy
        img = vbfs[scan_indexes].reshape(sdimy, sdimx)
        logger.debug("Applied calculated indexes and retrieved image")
        return img

    def get_blo_export_data(
        self,
        sdimx=None,
        sdimy=None,
        start_frame=None,
        end_frame=None,
        hyst=0,
        snakescan=True,
        crop=None,
    ):
        scan_indexes = self.calculate_scan_export_indexes(
            sdimx=sdimx,
            sdimy=sdimy,
            start_frame=start_frame,
            end_frame=end_frame,
            hyst=hyst,
            snakescan=snakescan,
            crop=crop,
        )
        logger.debug("Calculated scan indexes")
        if sdimx is None:
            sdimx = self.sdimx
        if sdimy is None:
            sdimy = self.sdimy
        try:
            imshap = self["ImageStream"]["Frame_000000"].shape
        except Exception:
            raise Exception("Could not find image size")
        if crop is not None:
            xmin, xmax, ymin, ymax = crop
            sdimx = xmax - xmin
            sdimy = ymax - ymin
        shape = (sdimx, sdimy, *imshap)
        return shape, scan_indexes

    def calculate_scan_export_indexes(
        self,
        sdimx=None,
        sdimy=None,
        start_frame=None,
        end_frame=None,
        hyst=0,
        snakescan=True,
        crop=None,
    ):
        """Calculate the indexes of the list of frames to consider for the
        scan or VBF"""
        try:
            rots = self["Scan"]["rotation_indexes"][:]
        except Exception:
            raise Exception(
                "No VBF information found in dataset, please "
                "calculate from TVIPS file."
            )
        logger.debug("Succesfully read rotator indexes")
        # set the scan info
        if sdimx is None:
            sdimx = self.sdimx
        if sdimy is None:
            sdimy = self.sdimy
        # check whether there are any dimensions
        if not isinstance(sdimx, int) or not isinstance(sdimy, int):
            raise Exception("No valid scan dimensions were found")
        # if a start frame is given, it's easy, we ignore rots
        if start_frame is not None:
            if end_frame is None:
                end_frame = start_frame + sdimx * sdimy - 1
            if end_frame >= self.total_frames:
                raise Exception("Final frame is out of bounds")
            if end_frame <= start_frame:
                raise Exception(
                    "Final frame index must be larger than first frame index"
                )
            if end_frame + 1 - start_frame != sdimx * sdimy:
                raise Exception("Number of custom frames does not match scan dimension")
            # just create an index array
            sel = np.arange(start_frame, end_frame + 1)
            sel = sel.reshape(sdimy, sdimx)
            # reverse correct even scan lines
            if snakescan:
                sel[::2] = sel[::2][:, ::-1]
            # hysteresis correction on even scan lines
            sel[::2] = np.roll(sel[::2], hyst, axis=1)

            # check for crop
            if crop is not None:
                logger.info("Cropping to: {}".format(crop))
                if all(i is not None for i in crop):
                    xmin, xmax, ymin, ymax = crop

                    if (
                        all(i >= 0 for i in (xmin, ymin))
                        and xmax < sdimx
                        and ymax < sdimy
                    ):
                        sel = sel[ymin:ymax, xmin:xmax]  # +1 to include final frame
                    else:
                        logger.warning(
                            "Aborting crop due to incorrect given dimensions: {}".format(
                                crop
                            )
                        )

            return sel.ravel()
        # if frames or not given, we must use our best guess and match
        # rotators
        else:
            try:
                rots = rots[self.start_frame : self.end_frame + 1]
            except Exception:
                raise Exception(
                    "No valid first or last scan frames detected, "
                    "must provide manual input"
                )
            # check whether sdimx*sdimy matches the final rotator index
            if not isinstance(self.final_rotator, int):
                raise Exception("No final rotator index found, " "can't align scan")
            if sdimx * sdimy != self.final_rotator:
                raise Exception(
                    "Scan dim x * scan dim y should match "
                    "the final rotator index if no custom "
                    "frames are specified"
                )
            indxs = np.zeros(sdimy * sdimx, dtype=int)
            prevw = 1
            for j, _ in enumerate(indxs):
                # find where the argument is j
                w = np.argwhere(rots == j + 1)
                if w.size > 0:
                    w = w[0, 0]
                    prevw = w
                else:
                    # move up if the rot index stays the same, otherwise copy
                    if prevw + 1 < len(rots):
                        if rots[prevw + 1] == rots[prevw]:
                            prevw = prevw + 1
                    w = prevw
                indxs[j] = w
            # just an array of indexes
            img = indxs.reshape(sdimy, sdimx)
            # reverse correct even scan lines
            if snakescan:
                img[::2] = img[::2][:, ::-1]
            # hysteresis correction on even scan lines
            img[::2] = np.roll(img[::2], hyst, axis=1)
            # add the start index
            return img.ravel() + self.start_frame
