import numpy as np
import os.path
from tifffile import FileHandle
import math
import h5py
from pathlib import Path
import re
from PyQt5.QtCore import QThread, pyqtSignal

import logging
from .imagefun import (normalize_convert, bin2, gausfilter,
                       medfilter)  # getElectronWavelength,

# Initialize the Logger
logger = logging.getLogger(__name__)

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
        self.convert_HDF5()
        self.finish.emit()

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
        self.scangroup.attrs["total_stream_frames"] = len(self.rotidxs)
        if self.end is not None and self.start is not None:
            self.scangroup.attrs[
                "ims_between_start_end"] = self.end-self.start

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


class hdf5Intermediate(h5py.File):
    """This class represents the intermediate hdf5 file handle"""
    def __init__(self, filepath, mode="r"):
        super().__init__(filepath, mode)
        (self.total_frames,
         self.start_frame,
         self.end_frame,
         self.final_rotator,
         dim, self.imdimx, self.imdimy) = self.get_scan_info()
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

    def get_vbf_image(self, sdimx=None, sdimy=None, start_frame=None,
                      end_frame=None, hyst=0, snakescan=True):
        # try to get the rotator data
        try:
            vbfs = self["Scan"]["vbf_intensities"][:]
        except Exception:
            raise Exception("No VBF information found in dataset, please "
                            "calculate from TVIPS file.")
        logger.debug("Succesfully imported vbf intensities")
        logger.debug("Now calculating scan indexes")
        scan_indexes = self.calculate_scan_export_indexes(
            sdimx=sdimx, sdimy=sdimy, start_frame=start_frame,
            end_frame=end_frame, hyst=hyst, snakescan=snakescan)
        logger.debug("Calculated scan indexes")
        if sdimx is None:
            sdimx = self.sdimx
        if sdimy is None:
            sdimy = self.sdimy
        img = vbfs[scan_indexes].reshape(sdimy, sdimx)
        logger.debug("Applied calculated indexes and retrieved image")
        return img

    def get_blo_export_data(self, sdimx=None, sdimy=None, start_frame=None,
                            end_frame=None, hyst=0, snakescan=True):
        scan_indexes = self.calculate_scan_export_indexes(
            sdimx=sdimx, sdimy=sdimy, start_frame=start_frame,
            end_frame=end_frame, hyst=hyst, snakescan=snakescan)
        logger.debug("Calculated scan indexes")
        if sdimx is None:
            sdimx = self.sdimx
        if sdimy is None:
            sdimy = self.sdimy
        try:
            imshap = self["ImageStream"]["Frame_000000"].shape
        except Exception:
            raise Exception("Could not find image size")
        shape = (sdimx, sdimy, *imshap)
        return shape, scan_indexes

    def calculate_scan_export_indexes(self, sdimx=None, sdimy=None,
                                      start_frame=None, end_frame=None,
                                      hyst=0, snakescan=True):
        """Calculate the indexes of the list of frames to consider for the
        scan or VBF"""
        try:
            rots = self["Scan"]["rotation_indexes"][:]
        except Exception:
            raise Exception("No VBF information found in dataset, please "
                            "calculate from TVIPS file.")
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
                end_frame = start_frame + sdimx*sdimy - 1
            if end_frame >= self.total_frames:
                raise Exception("Final frame is out of bounds")
            if end_frame <= start_frame:
                raise Exception("Final frame index must be larger than "
                                "first frame index")
            if end_frame+1-start_frame != sdimx*sdimy:
                raise Exception("Number of custom frames does not match "
                                "scan dimension")
            # just create an index array
            sel = np.arange(start_frame, end_frame+1)
            sel = sel.reshape(sdimy, sdimx)
            # reverse correct even scan lines
            if snakescan:
                sel[::2] = sel[::2][:, ::-1]
            # hysteresis correction on even scan lines
            sel[::2] = np.roll(sel[::2], hyst, axis=1)
            return sel.ravel()
        # if frames or not given, we must use our best guess and match
        # rotators
        else:
            try:
                rots = rots[self.start_frame:self.end_frame+1]
            except Exception:
                raise Exception("No valid first or last scan frames detected, "
                                "must provide manual input")
            # check whether sdimx*sdimy matches the final rotator index
            if not isinstance(self.final_rotator, int):
                raise Exception("No final rotator index found, "
                                "can't align scan")
            if sdimx*sdimy != self.final_rotator:
                raise Exception("Scan dim x * scan dim y should match "
                                "the final rotator index if no custom "
                                "frames are specified")
            indxs = np.zeros(sdimy*sdimx, dtype=int)
            prevw = 1
            for j, _ in enumerate(indxs):
                # find where the argument is j
                w = np.argwhere(rots == j+1)
                if w.size > 0:
                    w = w[0, 0]
                    prevw = w
                else:
                    # move up if the rot index stays the same, otherwise copy
                    if prevw+1 < len(rots):
                        if rots[prevw+1] == rots[prevw]:
                            prevw = prevw+1
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
            return img.ravel()+self.start_frame
