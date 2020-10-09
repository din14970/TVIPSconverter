import h5py
import numpy as np
import logging

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'HSPY'
description = \
    'The default file format for HyperSpy based on the HDF5 standard'
full_support = False
# Recognised file extension
file_extensions = ['hspy', 'hdf5']
default_extension = 0
# Writing capabilities
writes = True
version = "3.0"

PACKAGE_NAME = "pyxem"
PACKAGE_VERSION = "0.12.1"


def multiply(iterable):
    """Return product of sequence of numbers.
    Equivalent of functools.reduce(operator.mul, iterable, 1).
    >>> product([2**8, 2**30])
    274877906944
    >>> product([])
    1
    """
    prod = 1
    for i in iterable:
        prod *= i
    return prod


def get_signal_chunks(shape, dtype, signal_axes=None):
    """Function that claculates chunks for the signal, preferably at least one
    chunk per signal space.
    Parameters
    ----------
    shape : tuple
        the shape of the dataset to be sored / chunked
    dtype : {dtype, string}
        the numpy dtype of the data
    signal_axes: {None, iterable of ints}
        the axes defining "signal space" of the dataset. If None, the default
        h5py chunking is performed.
    """
    typesize = np.dtype(dtype).itemsize
    if signal_axes is None:
        return h5py._hl.filters.guess_chunk(shape, None, typesize)

    # largely based on the guess_chunk in h5py
    CHUNK_MAX = 1024 * 1024
    want_to_keep = multiply([shape[i] for i in signal_axes]) * typesize
    if want_to_keep >= CHUNK_MAX:
        chunks = [1 for _ in shape]
        for i in signal_axes:
            chunks[i] = shape[i]
        return tuple(chunks)

    chunks = [i for i in shape]
    idx = 0
    navigation_axes = tuple(i for i in range(len(shape)) if i not in
                            signal_axes)
    nchange = len(navigation_axes)
    while True:
        chunk_bytes = multiply(chunks) * typesize

        if chunk_bytes < CHUNK_MAX:
            break

        if multiply([chunks[i] for i in navigation_axes]) == 1:
            break
        change = navigation_axes[idx % nchange]
        chunks[change] = np.ceil(chunks[change] / 2.0)
        idx += 1
    return tuple(int(x) for x in chunks)


class hspyFileWriter(QThread):
    """Write a hyperspy file from an HDF5 without loading all data in memory"""
    increase_progress = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(self, fh, path_blo, shape, indexes,
                 scan_scale=5, diff_scale=1.075):
        QThread.__init__(self)
        self.fh = fh  # open hdf5 file in read mode
        self.path_blo = path_blo
        self.shape = shape  # shape of hypercube
        self.indexes = indexes  # indexes: right order of frames
        # scale seems unknown and arbitrarily determined in original code
        self.scan_scale = scan_scale
        self.diff_scale = diff_scale
        logger.debug("Initialized HSPYwriter")

    def run(self):
        self.convert_to_hspy()
        self.finish.emit()
        self.fh.close()  # close the hdf5 file

    def convert_to_hspy(self):
        """Convert hdf5 file to hyperspy compatible file"""
        with h5py.File(self.path_blo, mode='w') as f:
            f.attrs['file_format'] = "HyperSpy"
            f.attrs['file_format_version'] = version
            exps = f.create_group('Experiments')
            group_name = '__unnamed__'
            # experiment group
            expg = exps.create_group(group_name)
            expg.attrs["package"] = PACKAGE_NAME
            expg.attrs["package_version"] = PACKAGE_VERSION
            # add axes groups
            ax0 = expg.create_group('axis-0')
            ax0.attrs["name"] = "y"
            ax0.attrs["navigate"] = True
            ax0.attrs["offset"] = 0.0
            ax0.attrs["scale"] = float(self.scan_scale)
            ax0.attrs["size"] = int(self.shape[1])
            ax0.attrs["units"] = "nm"
            ax1 = expg.create_group('axis-1')
            ax1.attrs["name"] = "x"
            ax1.attrs["navigate"] = True
            ax1.attrs["offset"] = 0.0
            ax1.attrs["scale"] = float(self.scan_scale)
            ax1.attrs["size"] = int(self.shape[0])
            ax1.attrs["units"] = "nm"
            ax2 = expg.create_group('axis-2')
            ax2.attrs["name"] = "ky"
            ax2.attrs["navigate"] = False
            scale_ky = float(self.diff_scale)
            size_ky = int(self.shape[3])
            offset_ky = -scale_ky*(size_ky/2)
            ax2.attrs["offset"] = offset_ky/10
            ax2.attrs["scale"] = scale_ky/10
            ax2.attrs["size"] = size_ky
            ax2.attrs["units"] = "$A^{-1}$"
            ax3 = expg.create_group('axis-3')
            ax3.attrs["name"] = "kx"
            ax3.attrs["navigate"] = False
            scale_kx = float(self.diff_scale)
            size_kx = int(self.shape[2])
            offset_kx = -scale_kx*(size_kx/2)
            ax3.attrs["offset"] = offset_kx/10
            ax3.attrs["scale"] = scale_kx/10
            ax3.attrs["size"] = size_kx
            ax3.attrs["units"] = "$A^{-1}$"
            expg.create_group('learning_results')
            # metadata group
            metagroup = expg.create_group('metadata')
            generalgroup = metagroup.create_group("General")
            generalgroup.attrs["title"] = ""
            signalgroup = metagroup.create_group("Signal")
            signalgroup.attrs["binned"] = False
            signalgroup.attrs["record_by"] = 'image'
            signalgroup.attrs["signal_type"] = 'electron_diffraction'
            hyperspygroup = metagroup.create_group("Hyperspy")
            foldgroup = hyperspygroup.create_group("Folding")
            foldgroup.attrs["original_axes_manager"] = "_None_"
            foldgroup.attrs["original_shape"] = "_None_"
            foldgroup.attrs["signal_unfolded"] = False
            foldgroup.attrs["unfolded"] = False
            expg.create_group('original_metadata')
            # populate the data now
            c = f"{0}".zfill(6)
            ddtype = self.fh["ImageStream"][f"Frame_{c}"].dtype
            signal_axes = (3, 2)
            chunks = get_signal_chunks(self.shape, ddtype, signal_axes)
            maxshape = tuple(None for _ in self.shape)
            dset = expg.create_dataset(
                    name="data",
                    shape=(self.shape[1],
                           self.shape[0],
                           self.shape[3],
                           self.shape[2],),
                    dtype=ddtype,
                    chunks=chunks,
                    maxshape=maxshape,
                    compression='gzip',
                    shuffle=True,
                    )
            for j, indx in enumerate(self.indexes):
                c = f"{indx}".zfill(6)
                img = self.fh["ImageStream"][f"Frame_{c}"][:]
                xx = j % self.shape[0]
                yy = j // self.shape[0]
                dset[yy, xx, :, :] = img
                self.update_gui_progress(j)
                logger.debug(f"Wrote frame Frame_{c} to hyperspy file")

    def update_gui_progress(self, j):
        """If using the GUI update features with progress"""
        value = int((j+1)/len(self.indexes)*100)
        self.increase_progress.emit(value)
