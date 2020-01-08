import mrcfile
import tifffile
import numpy as np

import os
import os.path

from optparse import OptionParser

def _find_split(shape):
    #this assumes square image format
    #TODO: make that configurable
    frames = shape[0] // shape[1]
    return (frames, shape[1], shape[1])


def _create_destdir(destdir):
    try:
        os.stat(destdir)
    except:
        os.mkdir(destdir)

    return destdir


#TODO: add error handling, progress report etc.
def burst2mrc(files, keepfirst=False, destdir=".", force_overwrite=True):
    for file in files:
        if not str(file).lower().endswith(".tif"):
            continue

        #read in file
        tif = tifffile.TiffFile(file)

        try:
            if not tif.is_tvips:
                continue
        except AttributeError:
            print ('Consider updating your tifffile package, eg. by running "pip install tifffile --upgrade"')
            raise Exception("tifffile library outdated")


        basepath = _create_destdir(destdir)

        fn = os.path.join(basepath, os.path.basename(file))
        fn = fn[:-3] + "mrc"

        with mrcfile.new(name=fn, overwrite=force_overwrite) as mrc:
            tvips_metadata = tif.pages[0].tvips_metadata

            shape = _find_split((tvips_metadata['image_size_y'], tvips_metadata['image_size_x']))
            mrc.set_data(tif.asarray(memmap=True).reshape(shape)[0 if keepfirst else 1:])
            mrc.set_image_stack()

            #TODO: copy over header information, such as pixelsize etc.

            #TODO: understand the voxel size definition... http://mrcfile.readthedocs.io/en/latest/usage_guide.html#accessing-the-header-and-data
            mrc._set_voxel_size(tvips_metadata['pixel_size_x'], tvips_metadata['pixel_size_y'], 1)

def main():

    parser = OptionParser()

    parser.add_option("--keepfirst", dest="keepfirst", default=False, action="store_true",
                  help="Keep first frame of the stack (useful for normal mode)")
    parser.add_option("--destdir", dest="destdir", default=".",
                  help="Create mrc stacks in the specified directory")
    parser.add_option("--nooverwrite",
                  action="store_false", dest="overwrite", default=True,
                  help="Overwrite existing files")

    (options, args) = parser.parse_args()

    #call method
    burst2mrc(args, keepfirst=options.keepfirst, destdir=options.destdir, force_overwrite=options.overwrite)


if __name__ == "__main__":
    main()


# vim: sw=4 ts=4 et