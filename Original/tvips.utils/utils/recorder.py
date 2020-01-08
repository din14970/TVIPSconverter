import numpy as np
import os.path
from tifffile import FileHandle
import math
import gc

import logging
logging.basicConfig(level=logging.DEBUG)

from .imagefun import scale16to8, bin2, getElectronWavelength

TVIPS_RECORDER_GENERAL_HEADER = [
    ('size', 'u4'),
    ('version', 'u4'), #1 or 2
    ('dimx', 'u4'),
    ('dimy', 'u4'),
    ('bitsperpixel', 'u4'), #8 or 16
    ('offsetx', 'u4'),
    ('offsety', 'u4'),
    ('binx', 'u4'),
    ('biny', 'u4'),
    ('pixelsize', 'u4'), #nm, physical pixel size
    ('ht', 'u4'),
    ('magtotal', 'u4'),
    ('frameheaderbytes', 'u4'),
    ('dummy', 'S204'),

    ]

TVIPS_RECORDER_FRAME_HEADER = [
    ('num', 'u4'),
    ('timestamp', 'u4'), #seconds since 1.1.1970
    ('ms', 'u4'), #additional milliseconds to the timestamp
    ('LUTidx', 'u4'),
    ('fcurrent', 'f4'),
    ('mag', 'u4'),
    ('mode', 'u4'), #1 -> image 2 -> diff
    ('stagex', 'f4'),
    ('stagey', 'f4'),
    ('stagez', 'f4'),
    ('stagea', 'f4'),
    ('stageb', 'f4'),
    ('rotidx', 'u4'),
    ('temperature', 'f4'),
    ('objective', 'f4'),


    #for header version 2, some more data might be present
    ]

class Recorder(object):

    def __init__(self, filename, scalefunc=None, numframes=None, framerecord=None):
        assert os.path.exists(filename)
        assert filename.endswith(".tvips")

        self.general = None
        self.dtype = None
        self.frameHeader = list()
        self.frames = None
        self.scalefunc = scalefunc
        self.numframes = numframes

        self.frameshape = None

        #find numerical prefix
        part = int(filename[-9:-6])
        if part != 0:
            raise ValueError("Can only read video sequences starting with part 000")

        try:
            while True:
                fn = filename[:-9]+"{:03d}.tvips".format(part)

                if not os.path.exists(fn):
                    break

                frames, headers = self._readIndividualFile(fn, part)

                #merge memory efficient
                if (part==0):
                    self.frames = np.asarray(frames)
                else:
                    self.frames = np.append(self.frames, frames, axis=0)

                self.frameHeader.extend(headers)
                part += 1

        except StopIteration:
            pass

        print ("Read {} frames successfully".format(len(self.frames)))

    def _readIndividualFile(self, fn, part):
        logging.info("Reading {}".format(fn))

        frames = list()
        frame_headers = list()

        with open(fn, "rb") as f:
            fh = FileHandle(file=f)
            fh.seek(0)

            #read general header from first file
            if part == 0:
                self._readGeneral(fh)

            #respect desire not to read everything
            if self.numframes is not None:
                for j in range(self.numframes):
                    self._readFrame(fh, framerecord)
                raise StopIteration()
            else:
                #read all frames
                while fh.tell() < fh.size:
                    frame, header = self._readFrame(fh)

                    frames.append(frame)
                    frame_headers.append(header)
        return frames, frame_headers


    def _readGeneral(self, fh):
        self.general = fh.read_record(TVIPS_RECORDER_GENERAL_HEADER)

        self.dtype = np.uint8 if self.general.bitsperpixel == 8 else np.uint16
        self.frameshape = (self.general.dimx, self.general.dimy)

    def _readFrame(self, fh, record=None):
        inc = 12 if self.general.version == 1 else self.general.frameheaderbytes

        if self.general.version == 1:
            record = TVIPS_RECORDER_FRAME_HEADER

        if record is None:
            record = TVIPS_RECORDER_FRAME_HEADER
            if inc > 12:
                pass
                #print("Warning: Custom Frame Header detected. Please supply matching record definition.")

        dt = np.dtype(record)

        #make sure the record consumes less bytes than reported in the main header
        assert inc >= dt.itemsize

        #read header
        header = fh.read_record(record)
        logging.debug("Starting frame read (pos: {}): {}".format(fh.tell(), header))
        skip = inc - dt.itemsize

        fh.seek(skip, 1)

        #read frame
        frame = np.fromfile(fh,
                    count=self.general.dimx*self.general.dimy,
                    dtype=self.dtype
                    )
        frame.shape = (self.general.dimx, self.general.dimy)
        logging.debug("Starting filters")

        if self.scalefunc is not None:
            frame = self.scalefunc(frame).astype(np.uint8)

        logging.debug("Finished filters")
        return frame, header

    def toarray(self):
        return np.asarray(self.frames)




def main_spill_to_directory():
    import sys
    assert len(sys.argv) == 3

    #first arg: .tvips file
    #second arg: directory to spill the .tif files

    import os
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])

    import tifffile

    print("Reading in file {}.".format(sys.argv[1]))
    rec = Recorder(sys.argv[1])

    numframes = len(rec.frames)

    amount_of_digits = len(str(numframes-1))

    logging.debug("Start writing individual tif files to {}".format(sys.argv[2]))

    filename = "frame_{:0" + str(amount_of_digits) + "d}.tif"
    filename = os.path.join(sys.argv[2], filename)

    for i, frame in enumerate(rec.frames):

        tifffile.imsave(filename.format(i), frame)

    print ("Done saving {:d} frames to {}.".format(i+1, sys.argv[2]))

def main():
    import sys
    import argparse
    import tifffile

    from enum import Enum

    class OutputTypes(Enum):
        IndividualTiff="Individual"
        TiffStack = "TiffStack"
        Blockfile = "blo"
        VirtualBF = "VirtualBF"

        def __str__(self):
            return self.value

    def correct_column_offsets(image, thresholdmin=0, thresholdmax=30, binning=1):
        pixperchannel = int(128 / binning)

        if (128.0/binning != pixperchannel):
            print("Can't figure out column offset dimension")
            return image

        numcol = int(image.shape[0] / 128 * binning)


        #this is too complicated for me to write in just one expression - so use a loop
        imtemp = image.reshape((image.shape[0], pixperchannel, numcol))
        offsets = []
        for j in range(numcol):
            channel = imtemp[:,j,:]
            pdb.set_trace()
            mask = np.bitwise_and (channel < thresholdmax, channel >= thresholdmin)
            value = np.mean(channel[mask])
            offsets.append(value)

        #apply offset correction to images
        offsets = np.array(offsets)
        return (imtemp - offsets[np.newaxis, :]).reshape(image.shape)

    def virtual_bf_mask(image, centeroffsetpx=(0.0), radiuspx=10):

        xx, yy = np.meshgrid(np.arange(image.shape[0], dtype=np.float), np.arange(image.shape[1], dtype=np.float))
        xx -= 0.5 * image.shape[0] + centeroffsetpx[0]
        yy -= 0.5 * image.shape[1] + centeroffsetpx[1]

        mask = np.hypot(xx,yy)<radiuspx
        return mask



    parser = argparse.ArgumentParser(description='Process .tvips recorder format')

    parser.add_argument('--otype', type=OutputTypes, choices=list(OutputTypes), help='Output format')
    parser.add_argument("--numframes", type=int, default=None, help="Limit data to the first n frames")
    parser.add_argument("--binning", type=int, default=None, help="Bin data")

    parser.add_argument("--dumpheaders", action="store_true", default=False, help="Dump headers")

    parser.add_argument('--depth', choices=("uint8", "uint16", "int16"), default=None)
    parser.add_argument('--linscale', help="Scale 16 bit data linear to 8 bit using the given range. Eg. 100-1000. Default: min, max", default=None)
    parser.add_argument('--coffset', action='store_true', default=False)

    #Virtual BF/blo options
    parser.add_argument('--vbfcenter', default='0.0x0.0', help='Offset to center of Zero Order Beam')
    parser.add_argument('--vbfradius', default=10.0, type=float, help='Integration disk radius')
    parser.add_argument('--dimension', help='Output dimensions, default: sqrt(#images)^2')
    parser.add_argument('--rotator', action="store_true", help="Pick only valid rotator frames")
    parser.add_argument('--hysteresis', default=0, type=int, help='Move every second row by n pixel')
    parser.add_argument('--postmag', default=1.0, type=float, help="Apply a mag correction")

    parser.add_argument('--skip', default=0, type=int, help='Skip # images at the beginning')
    parser.add_argument('--truncate', default=0, type=int, help='Truncate # images at the end')

    parser.add_argument('input', help="Input filename, must be _000.tvips")
    parser.add_argument('output', help="Output dir or output filename")

    opts = parser.parse_args()

    def determine_recorder_image_dimension():
        #image dimension
        xdim, ydim = 0, 0
        if (opts.dimension is not None):
            xdim, ydim = list(map(int, opts.dimension.split('x')))
        else:
            dim = math.sqrt(len(rec.frames))
            if not dim == int(dim):
                raise ValueError("Can't determine correct image dimensions, please supply values manually (--dimension)")
            xdim, ydim = dim, dim
            print("Determining image dimensions to {}x{}".format(xdim, ydim))

        return xdim, ydim

    #read in file
    assert (os.path.exists(opts.input))

    scalefunc = None

    if opts.linscale is not None or opts.binning is not None:
        binfunc = lambda x: x
        if opts.binning:
            logging.info("Binning data by {:d}".format(opts.binning))
            binfunc = lambda x: bin2(x, opts.binning)

        if opts.linscale:
            min, max = map(float, opts.linscale.split('-'))
            scalefunc = lambda x: scale16to8(binfunc(x), min, max)
            logging.info("Mapping range of {}-{} to 0-255".format(min, max))

    #read tvips file
    rec = Recorder(opts.input, scalefunc=scalefunc, numframes=opts.numframes)

    #truncate frames
    if (opts.skip != 0 and opts.truncate != 0):
        rec.frames = rec.frames[opts.skip:-opts.truncate]
        rec.frameHeader = rec.frameHeader[opts.skip:-opts.truncate]
    elif (opts.skip != 0 and opts.truncate == 0):
        rec.frames = rec.frames[opts.skip:]
        rec.frameHeader = rec.frameHeader[opts.skip:]
    elif (opts.skip == 0 and opts.truncate != 0):
        rec.frames = rec.frames[:-opts.truncate]
        rec.frameHeader = rec.frameHeader[:-opts.truncate]
    else:
        pass #nothing requested

    if (opts.dumpheaders):
        print("General:\n{}\nFrame:\n{}".format(rec.general, rec.frameHeader))

    if (opts.rotator):
        previous=0x43 #some unlikely initial value
        start = None
        end = None
        i=0

        numframes = None

        if opts.dimension is not None:
            xdim, ydim = list(map(int, opts.dimension.split('x')))
            numframes = xdim*ydim

        for i, fh in enumerate(rec.frameHeader):
            logging.debug("Frame: {:05d} Index: {:05d}".format(i, fh['rotidx']))
            if start is None and end is None:
                #set start idx
                if fh['rotidx'] == previous + 1: #first consecutive frame idx
                    start = i - 1 #TODO: check for > 0
                    logging.info("Found start at {}".format(start))
                    continue

            if start is not None:
                if numframes is None:
                    #find end idx
                    if fh['rotidx'] == previous:
                        break

                else:
                    logging.info("Taking {} frames based on given dimensions".format(numframes))

                    #manipulate i, not end as it will be assigned outside loop
                    i = start + numframes

                    logging.debug("Start: {:05d} End: {:05d}".format(start, i))

                    if (len(rec.frameHeader) <= i):
                        logging.error("Too few records in dataset for the given dimensions")

                    break




            previous = fh['rotidx']

        end = i
        logging.info ("Found end at {}".format(end))

        #remove uninteresting data
        rec.frames = rec.frames[start:end]
        rec.frameHeader = rec.frameHeader[start:end]
        logging.info ("Found {} frames in set".format(len(rec.frames)))

        #rec.frames = np.asarray(rec.frames)
        #rec.frameHeader = np.asarray(rec.frameHeader)

    if (opts.coffset):
        rec.frames = map(correct_column_offsets, rec.frames)

    if (opts.depth is not None):
        dtype = np.dtype(opts.depth) #parse dtype
        logging.debug("Mapping data to {}...".format(opts.depth))
        #rec.frames = list(map(lambda x: x.astype(dtype), rec.frames))
        rec.frames = [x.astype(dtype) for x in rec.frames]
        logging.debug("Done Mapping")

    if (opts.otype == OutputTypes.IndividualTiff):
            numframes = len(rec.frames)
            amount_of_digits = len(str(numframes-1))

            print("Start writing individual tif files to {}".format(opts.output))
            if not os.path.exists(opts.output):
                os.mkdir(opts.output)

            filename = "frame_{:0" + str(amount_of_digits) + "d}.tif"
            filename = os.path.join(opts.output, filename)

            for i, frame in enumerate(rec.frames):
                tifffile.imsave(filename.format(i), frame)
    elif (opts.otype == OutputTypes.TiffStack):
        tifffile.imsave(opts.output, rec.toarray())

    elif (opts.otype == OutputTypes.VirtualBF):
        xdim, ydim = determine_recorder_image_dimension()
        oimage = np.zeros((xdim*ydim), dtype=rec.frames[0].dtype)

        #ZOB center offset
        zoboffset = list(map(float, opts.vbfcenter.split('x')))

        #generate
        mask = virtual_bf_mask(rec.frames[0], zoboffset, opts.vbfradius)
        for i, frame in enumerate(rec.frames[:xdim*ydim]):
            oimage[i] = frame[mask].mean()

        oimage.shape=(xdim, ydim)

        #correct for meander scanning of rotator
        if (opts.rotator):
            oimage[::2] = oimage[::2][:,::-1]

            #correct for hysteresis
            if opts.hysteresis != 0:
                logging.info("Correcting for hysteresis...")
                oimage[::2] = np.roll(oimage[::2], opts.hysteresis, axis=1)
                logging.info("Rescaling to valid area...")
                oimage = oimage[:, opts.hysteresis:]


        logging.info("Writing out image")
        tifffile.imsave(opts.output, oimage)

    elif (opts.otype == OutputTypes.Blockfile):
        from . import blockfile
        xdim, ydim = determine_recorder_image_dimension()

        gc.collect()
        arr = rec.frames


        if (len(arr) != xdim * ydim):

            #extend it to the requested dimensions
            missing = xdim*ydim - len(arr)
            arr = np.concatenate((arr, missing * [np.zeros_like(arr[0]),]))

            logging.info("Data set filled up with {} frames for matching requested dimensions".format(missing))

        arr.shape=(xdim, ydim, *arr[0].shape)

        #reorder meander
        arr[::2] = arr[::2][:,::-1]

        #np.savez_compressed("c:\\temp\\dump.npz", data=arr)

        #TODO: check whether valid
        #apply hysteresis correction
        if opts.hysteresis != 0:
            logging.info("Correcting for hysteresis...")
            arr[::2] = np.roll(arr[::2], opts.hysteresis, axis=1)
            logging.info("Rescaling to valid area...")
            #arr = arr[:, opts.hysteresis:]

        #write out as tiffstack for now, later blo file with good header
        #tifffile.imsave(opts.output, arr, bigtiff=True)

        #calculate header flags
        totalbinning = opts.binning * rec.general['binx']
        wl = getElectronWavelength(1000.0 * rec.general['ht'])
        pxsize = 1e-9 * rec.general['pixelsize']

        #wl in A * cl in cm * px per meter
        ppcm = wl*1e10 * rec.general['magtotal'] / (pxsize*totalbinning*opts.postmag)

        blockfile.file_writer_array(opts.output, arr, 5, 1.075,
                Camera_length=100.0*rec.general['magtotal'],
                Beam_energy=rec.general['ht']*1000,
                Distortion_N01=1.0, Distortion_N09=1.0,
                Note="Cheers from TVIPS!")
    else:
        raise ValueError("No output type specified (--otype)")


if __name__ == "__main__":
    main()
