import logging
from . import imagefun
from pathlib import Path
import tifffile

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class TiffFileWriter(QThread):
    """Write a blo file from an HDF5 without loading all data in memory"""
    increase_progress = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(self, fh, indexes, dtype, pre, pos):
        QThread.__init__(self)
        self.fh = fh  # open hdf5 file in read mode
        self.dtype = dtype
        self.indexes = indexes  # indexes: right order of frames
        # naming of the files
        self.pre = pre
        self.pos = pos
        logger.debug("Initilized tiff writer")

    def run(self):
        self.write_tiffs()
        self.finish.emit()
        self.fh.close()  # close the hdf5 file

    def write_tiffs(self):
        for j, i in enumerate(self.indexes):
            c = f"{i}".zfill(6)
            img = self.fh["ImageStream"][f"Frame_{c}"][:]
            img = imagefun.normalize_convert(img, dtype=self.dtype)
            fname = str(Path(f"{self.pre}_{c}{self.pos}"))
            tifffile.imsave(fname, img)
            self.update_gui_progress(j)
            logger.debug(f"Saved frame {c} as tiff")
        logger.debug(f"Finished writing tiffs")

    def update_gui_progress(self, j):
        """If using the GUI update features with progress"""
        value = int((j+1)/len(self.indexes)*100)
        self.increase_progress.emit(value)
