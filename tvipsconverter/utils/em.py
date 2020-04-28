import numpy as np

import os
import os.path
import glob

from optparse import OptionParser

from tifffile import FileHandle


EM_FORMAT_HEADER = [
        ('machine', 'u1'),
        ('dummy0', 'u2'),
        ('datatype', 'u1'),

        ('xdim', 'u4'),
        ('ydim', 'u4'),
        ('zdim', 'u4'),

        ('comment', 'S80'),
        ('data', '40i4'),
        ('userdata', 'S28'),

        ('fftype', 'u4'),
        ('ffcameragroup', 'u4'),
        ('ffoffsetx', 'u4'),
        ('ffoffsety', 'u4'),
        ('ffdimx', 'u4'),
        ('ffdimy', 'u4'),
        ('ffbinx', 'u4'),
        ('ffbiny', 'u4'),

        ('ffgainindex', 'u4'),
        ('ffspeedindex', 'u4'),
        ('ffcameramode', 'u4'),
        ('ffspecial', 'u4'),
        ('ffexposuretime', 'u4'),
        ('ffmean', 'u4'),
        ('ffcreationtime', 'u4'), #seconds since 1.1.1970

        ('dummy1', 'S168'),
         ]

EM_DTYPE = [
    ValueError("Invalid data type"),
    np.uint8,
    np.int16,
    np.uint16,
    ]



class TVIPS_EM(object):
    image = None
    header = None

    def __init__(self, file):
        f = FileHandle(file=file)
        self.header = f.read_record(EM_FORMAT_HEADER)
        self.image = np.fromfile(f, dtype=EM_DTYPE[self.header.datatype])
        f.close()

        self.image.shape = (self.header.ffdimx, self.header.ffdimy)

        #watch out: F*.fff files have an offset of 500 (default) or another value

def main():

    parser = OptionParser()

    parser.add_option("--mode", dest="mode", default=None, type="int",
                  help="Set Readout Mode")

    parser.add_option("--speed", dest="speed", default=None, type="int",
                  help="Set Speed Index")

    parser.add_option("--special", dest="special", default=None, type="int",
                  help="Set Special Flag")
    parser.add_option("--mirrory",
                  action="store_true", dest="mirrory", default=False,
                  help="Mirror Y Axis")
    parser.add_option("--notinplace", action="store_false", dest="inplace", default=True,
                      help="Manipulate images in place")

    (options, args) = parser.parse_args()

    #args: filenames

    if 0 == len(args):
        parser.print_help()

    #teach windows some unix tricks
    if args[0].find("*"):
        args = glob.glob(args[0])

    for file in args:
        #manipulate files in-place
        if not os.path.exists(file):
            print ("Skipping {}".format(file))
            continue

        if not file.endswith((".em", ".fff")):
            continue

        print ("Working on {}".format(file))

        f = FileHandle(file=file)
        header = f.read_record(EM_FORMAT_HEADER)
        image = np.fromfile(f, dtype=EM_DTYPE[header.datatype])

        f.close()

        image.shape = (header.ffdimx, header.ffdimy)

        #do file manipulation
        if options.mode is not None:
            header.ffcameramode = options.mode

        if options.speed is not None:
            header.ffspeedindex = options.speed

        #print ("vals:  {:d} {:d}".format(image[0,0], image[0, -1]))

        if options.mirrory:
            print("Flipping in Y axis")
            image = np.fliplr(image)

        if options.special is not None:
            header.ffspecial = options.special

        #write back file

        if options.inplace:
            fname = file
        else:
            dir = os.path.dirname(file)
            fn = "".join(str.split('.', os.path.basename(file))[:-1])
            suff = str.split('.', os.path.basename(file))[-1]

            fname = fn + "_mod." + suff

        with open(fname, "wb") as newfile:
            newfile.write(header.tobytes())
            newfile.write(image.tobytes())


if __name__ == "__main__":
    main()


# vim: sw=4 ts=4 et
