from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGraphicsScene)
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as
                                                FigureCanvas)
from pathlib import Path
import logging
from time import sleep
import numpy as np
import os

from .utils import recorder as rec
from .utils import blockfile as blf
from .utils import tiffexport as tfe

logging.basicConfig(level=logging.DEBUG)
sys.path.append(".")

logger = logging.getLogger(__name__)
# import the UI interface
rawgui, Window = uic.loadUiType(str(Path(__file__).parent.absolute()) +
                                "/widget_2.ui")


class External(QThread):
    """
    Runs a counter thread.
    """
    countChanged = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(self, fin):
        QThread.__init__(self)
        self.fin = fin

    def run(self):
        count = 0
        while count < self.fin:
            count += 1
            sleep(0.1)
            self.countChanged.emit(count)
        self.finish.emit()


class ConnectedWidget(rawgui):
    """Class connecting the gui elements to the back-end functionality"""
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setupUi(window)

        self.original_preview = None
        self.path_preview = None

        # has a valid hdf5 file been selected or not
        self.valid_hdf5 = False

        # data storage for preview (only need figure)
        self.fig_prev = None

        # data storage for vbf preview
        self.vbf_data = None
        self.vbf_sets = None  # settings
        self.fig_vbf = None
        self.vbf_im = None

        self.connectUI()

    def connectUI(self):
        # initial browse button
        self.pushButton.clicked.connect(self.open_tvips_file)

        # update button on the preview
        self.pushButton_4.clicked.connect(self.updatePreview)
        #
        # # execute the conversion command
        self.pushButton_3.clicked.connect(self.get_hdf5_path)
        # shitty workaround for saving to hdf5 with updated gui
        self.pushButton_6.clicked.connect(self.write_to_hdf5)
        self.lineEdit_4.textChanged.connect(self.auto_read_hdf5)
        # self.pushButton_3.clicked.connect(self.exportFiles)
        # exporting the preview
        self.pushButton_2.clicked.connect(self.export_preview)
        # browse to select a file
        self.pushButton_8.clicked.connect(self.select_hdf5_file)
        # auto update custom scan dimensions (spinbox 16)
        self.spinBox.valueChanged.connect(self.update_final_frame)
        self.spinBox_2.valueChanged.connect(self.update_final_frame)
        self.spinBox_15.valueChanged.connect(self.update_final_frame)
        self.checkBox_2.stateChanged.connect(self.update_final_frame)
        # create a preview of the vbf
        self.pushButton_10.clicked.connect(self.update_vbf)
        # connecting the horizontal sliders
        self.horizontalSlider.sliderReleased.connect(self.update_levels_vbf)
        self.horizontalSlider_2.sliderReleased.connect(self.update_levels_vbf)
        # saving the VBF
        self.pushButton_7.clicked.connect(self.export_vbf)
        # # deactivate part of the gui upon activation
        # browsing blo file
        self.pushButton_9.clicked.connect(self.get_blo_path)
        # start blo conversion
        self.pushButton_5.clicked.connect(self.write_to_blo)
        # select tiff file
        self.pushButton_12.clicked.connect(self.get_tiff_path)
        # export tiff files
        self.pushButton_11.clicked.connect(self.export_tiffs)
        # self.checkBox_8.stateChanged.connect(self.updateActive)
        # self.checkBox_3.stateChanged.connect(self.updateActive)
        # self.checkBox_7.stateChanged.connect(self.updateActive)
        # self.checkBox_5.stateChanged.connect(self.updateActive)
        # self.checkBox_4.stateChanged.connect(self.updateActive)
        # self.checkBox_6.stateChanged.connect(self.updateActive)
        # self.updateActive()

    def export_preview(self):
        try:
            if self.fig_prev is None:
                raise Exception("Must first create preview")
            path = self.saveFileBrowser("PNG (*.png)")
            if path is None:
                raise Exception("No valid file selected")
            self.fig_prev.savefig(path)
            self.update_line(self.statusedit, "Succesfully saved preview.")
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def export_vbf(self):
        try:
            if self.fig_vbf is None:
                raise Exception("Must first create VBF preview")
            path = self.saveFileBrowser("PNG (*.png)")
            if path is None:
                raise Exception("No valid file selected")
            self.fig_vbf.savefig(path)
            self.update_line(self.statusedit, "Succesfully saved VBF.")
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def update_levels_vbf(self):
        if self.vbf_data is not None:
            vmin = self.horizontalSlider.value()
            vmax = self.horizontalSlider_2.value()
            mn = self.vbf_data.min()
            mx = self.vbf_data.max()
            unit = (mx-mn)/100
            climmin = mn+vmin*unit
            climmax = mn+(vmax+1)*unit
            try:
                self.vbf_im.set_clim(climmin, climmax)
                canvas = FigureCanvas(self.fig_vbf)
                canvas.draw()
                scene = QGraphicsScene()
                scene.addWidget(canvas)
                self.graphicsView_3.setScene(scene)
                self.graphicsView_3.fitInView(scene.sceneRect())
                self.repaint_widget(self.graphicsView_3)
            except Exception as e:
                logger.debug(f"Error: {e}")

    def save_vbf_to_hdf5(self):
        pass

    def update_final_frame(self):
        if self.checkBox_2.checkState():
            # we use self defined size
            start = self.spinBox_15.value()
            frms = self.spinBox.value()*self.spinBox_2.value()
            self.spinBox_16.setValue(start+frms-1)
        else:
            # we use auto-size
            start = self.spinBox_15.value()
            frms = self.lineEdit_11.text()
            try:
                dim = np.sqrt(int(frms))
                self.spinBox_16.setValue(start+dim**2-1)
            except Exception:
                self.spinBox_16.setValue(0)

    def threadCheck(self):
        self.calc = External(50)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.finish.connect(self.done)
        self.calc.start()
        self.window.setEnabled(False)

    def onCountChanged(self, value):
        self.progressBar.setValue(value)

    def open_tvips_file(self):
        path = self.openFileBrowser("TVIPS (*.tvips)")
        # check if it's a valid file
        if path:
            try:
                rec.Recorder.valid_first_tvips_file(path)
                self.lineEdit.setText(path)
                self.statusedit.setText("Selected tvips file")
            except Exception as e:
                self.lineEdit.setText("")
                self.statusedit.setText(str(e))

    def openFileBrowser(self, fs):
        path, okpres = QFileDialog.getOpenFileName(caption="Select file",
                                                   filter=fs)
        if okpres:
            return str(Path(path))

    def saveFileBrowser(self, fs):
        path, okpres = QFileDialog.getSaveFileName(caption="Select file",
                                                   filter=fs)
        if okpres:
            return str(Path(path))

    def openFolderBrowser(self):
        path = QFileDialog.getExistingDirectory(caption="Choose directory")
        if path:
            return str(Path(path))

    def read_modsettings(self):
        path = self.lineEdit.text()
        improc = {
            "useint": self.checkBox_8.checkState(),
            "whichint": self.spinBox_11.value(),
            "usebin": self.checkBox_5.checkState(),
            "whichbin": self.spinBox_3.value(),
            "usegaus": self.checkBox_3.checkState(),
            "gausks": self.spinBox_6.value(),
            "gaussig": self.doubleSpinBox.value(),
            "usemed": self.checkBox_7.checkState(),
            "medks": self.spinBox_10.value(),
            "usels": self.checkBox_4.checkState(),
            "lsmin": self.spinBox_7.value(),
            "lsmax": self.spinBox_8.value(),
            "usecoffset": self.checkBox.checkState()
        }
        vbfsettings = {
            "calcvbf": self.checkBox_10.checkState(),
            "vbfrad": self.spinBox_12.value(),
            "vbfxoffset": self.spinBox_13.value(),
            "vbfyoffset": self.spinBox_14.value()
        }
        return path, improc, vbfsettings

    def updatePreview(self):
        """Read the first image from the file and create a preview"""
        # read the gui info
        path, improc, vbfsettings = self.read_modsettings()
        # create one image properly processed
        try:
            if not path:
                raise Exception("A TVIPS file must be selected!")
            # get and calculate the image. Also get old image and new image
            # size. only change original image if there is none or the path
            # has changed
            if (self.original_preview is None) or (self.path_preview != path):
                self.update_line(self.statusedit, "Extracting frame...")
                self.original_preview = rec.getOriginalPreviewImage(
                                                path, improc=improc,
                                                vbfsettings=vbfsettings)
            # update the path
            self.path_preview = path
            ois = self.original_preview.shape
            filterframe = rec.filter_image(self.original_preview, **improc)
            nis = filterframe.shape
            # check if the VBF aperture fits in the frame
            if vbfsettings["calcvbf"]:
                midx = nis[1]//2
                midy = nis[0]//2
                xx = vbfsettings["vbfxoffset"]
                yy = vbfsettings["vbfyoffset"]
                rr = vbfsettings["vbfrad"]
                if (midx+xx-rr < 0 or
                   midx+xx+rr > nis[1] or
                   midy+yy-rr < 0 or
                   midy+yy-rr > nis[0]):
                    raise Exception("Virtual bright field aperture out "
                                    "of bounds")
            # plot the image and the circle over it
            if self.fig_prev is not None:
                plt.close(self.fig_prev)
            self.fig_prev = plt.figure(frameon=False,
                                       figsize=(filterframe.shape[1]/100,
                                                filterframe.shape[0]/100))
            canvas = FigureCanvas(self.fig_prev)
            ax = plt.Axes(self.fig_prev, [0., 0., 1., 1.])
            ax.set_axis_off()
            self.fig_prev.add_axes(ax)
            ax.imshow(filterframe, cmap="Greys_r")
            if vbfsettings["calcvbf"]:
                xoff = vbfsettings["vbfxoffset"]
                yoff = vbfsettings["vbfyoffset"]
                circ = Circle((filterframe.shape[1]//2+xoff,
                               filterframe.shape[0]//2+yoff),
                              vbfsettings["vbfrad"],
                              color="red",
                              alpha=0.5)
                ax.add_patch(circ)
            canvas.draw()
            scene = QGraphicsScene()
            scene.addWidget(canvas)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect())
            self.repaint_widget(self.graphicsView)
            self.update_line(self.statusedit, "Succesfully created preview.")
            self.update_line(self.lineEdit_8, f"Original: {ois[0]}x{ois[1]}. "
                                              f"New: {nis[0]}x{nis[1]}.")
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")
            # empty the preview
            self.update_line(self.lineEdit_8, "")
            scene = QGraphicsScene()
            self.graphicsView.setScene(scene)
            self.repaint_widget(self.graphicsView)
            self.original_preview = None
            self.path_preview = None

    def repaint_widget(self, widget):
        widget.hide()
        widget.show()

    def update_line(self, line, string):
        line.setText(string)
        line.hide()
        line.show()

    def get_hdf5_path(self):
        # open a savefile browser
        try:
            # read the gui info
            (self.inpath, self.improc,
             self.vbfsettings) = self.read_modsettings()
            if not self.inpath:
                raise Exception("A TVIPS file must be selected!")
            self.oupath = self.saveFileBrowser("HDF5 (*.hdf5)")
            if not self.oupath:
                raise Exception("No valid HDF5 file path selected")
            self.lineEdit_2.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def select_hdf5_file(self):
        # open an open file browser
        try:
            # read the gui info
            hdf5path = self.openFileBrowser("HDF5 (*.hdf5)")
            if not hdf5path:
                raise Exception("No valid HDF5 file path selected")
            self.lineEdit_4.setText(hdf5path)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def get_blo_path(self):
        # open a savefile browser
        try:
            self.oupath = self.saveFileBrowser("BLO (*.blo)")
            if not self.oupath:
                raise Exception("No valid BLO file path selected")
            self.lineEdit_7.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def get_tiff_path(self):
        # open a savefile browser
        try:
            self.oupath = self.saveFileBrowser("tiff (*.tiff)")
            if not self.oupath:
                raise Exception("No valid tiff file path selected")
            self.lineEdit_9.setText(self.oupath)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done(self):
        self.window.setEnabled(True)

    def write_to_hdf5(self):
        try:
            (self.inpath, self.improc,
             self.vbfsettings) = self.read_modsettings()
            if not self.inpath:
                raise Exception("A TVIPS file must be selected!")
            self.oupath = self.lineEdit_2.text()
            if not self.oupath:
                raise Exception("No valid HDF5 file path selected")
            # read the gui info
            self.update_line(self.statusedit, f"Exporting to {self.oupath}")
            path = self.inpath
            opath = self.oupath
            improc = self.improc
            vbfsettings = self.vbfsettings
            self.get_thread = rec.Recorder(path,
                                           improc=improc,
                                           vbfsettings=vbfsettings,
                                           outputpath=opath)
            self.get_thread.increase_progress.connect(self.increase_progbar)
            self.get_thread.finish.connect(self.done_hdf5export)
            self.get_thread.start()
            self.window.setEnabled(False)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done_hdf5export(self):
        self.window.setEnabled(True)
        # also update lines in the second pannel
        self.update_line(self.statusedit,
                         "Succesfully exported to HDF5")
        # don't auto update, the gui may be before the file exists
        # self.update_line(self.lineEdit_4, self.lineEdit_2.text())

    def auto_read_hdf5(self):
        """Update HDF5 field info if lineEdit_4 (path) is changed"""
        try:
            f = rec.hdf5Intermediate(self.lineEdit_4.text())
            tot, star, en, roti, dim, imdimx, imdimy = f.get_scan_info()
            if tot is not None:
                self.update_line(self.lineEdit_3, str(tot))
            else:
                self.update_line(self.lineEdit_3, "?")
            if star is not None:
                self.update_line(self.lineEdit_5, str(star))
            else:
                self.update_line(self.lineEdit_5, "?")
            if en is not None:
                self.update_line(self.lineEdit_6, str(en))
            else:
                self.update_line(self.lineEdit_6, "?")
            if roti is not None:
                self.update_line(self.lineEdit_11, str(roti))
            else:
                self.update_line(self.lineEdit_11, "?")
            if dim is not None:
                self.update_line(self.lineEdit_12,
                                 f"{str(int(dim))}x{str(int(dim))}")
            else:
                self.update_line(self.lineEdit_12, "?")
            self.update_line(self.lineEdit_13,
                             f"{str(imdimx)}x{str(imdimy)}")
            self.update_final_frame()
            f.close()
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")
            self.update_line(self.lineEdit_3, "")
            self.update_line(self.lineEdit_5, "")
            self.update_line(self.lineEdit_6, "")
            self.update_line(self.lineEdit_11, "")
            self.update_line(self.lineEdit_12, "")
            self.update_line(self.lineEdit_12, "")

    def update_vbf(self):
        """Calculate the virtual bf """
        path_hdf5 = self.lineEdit_4.text()
        try:
            # check if an hdf5 file is selected
            if not path_hdf5:
                raise Exception("No valid HDF5 file selected!")
            # try to read the info from the file
            f = rec.hdf5Intermediate(path_hdf5)
            # none or 0 means default
            start_frame = None
            end_frame = None
            sdimx = None
            sdimy = None
            hyst = 0
            snakescan = True
            # overwrite standard info depending on gui
            if self.checkBox_2.checkState():
                # use custom scanning
                sdimx = self.spinBox.value()
                sdimy = self.spinBox_2.value()
            if self.checkBox_11.checkState():
                # use custom frames
                start_frame = self.spinBox_15.value()
                end_frame = self.spinBox_16.value()
            # use hysteresis or not
            if self.checkBox_6.checkState():
                hyst = self.spinBox_9.value()
            # use snake scan or not
            if self.checkBox_12.checkState():
                snakescan = True
            else:
                snakescan = False
            # calculate the image
            logger.debug(f"We try to create a VBF image with data: "
                         f"S.F. {start_frame}, E.F. {end_frame}, "
                         f"Dims: x {sdimx} y {sdimy},"
                         f"hyst: {hyst}")
            self.vbf_data = f.get_vbf_image(sdimx, sdimy, start_frame,
                                            end_frame, hyst, snakescan)
            logger.debug("Succesfully created the VBF array")
            # save the settings for later storage
            self.vbf_sets = {"start_frame": start_frame,
                             "end_frame": end_frame,
                             "scan_dim_x": sdimx,
                             "scan_dim_y": sdimy,
                             "hysteresis": hyst,
                             "winding_scan": snakescan}
            # plot the image and store it for further use. First close prior
            # image
            if self.fig_vbf is not None:
                plt.close(self.fig_vbf)
            self.fig_vbf = plt.figure(frameon=False,
                                      figsize=(self.vbf_data.shape[1]/100,
                                               self.vbf_data.shape[0]/100))
            canvas = FigureCanvas(self.fig_vbf)
            ax = plt.Axes(self.fig_vbf, [0., 0., 1., 1.])
            ax.set_axis_off()
            self.fig_vbf.add_axes(ax)
            self.vbf_im = ax.imshow(self.vbf_data, cmap="plasma")
            canvas.draw()
            scene = QGraphicsScene()
            scene.addWidget(canvas)
            self.graphicsView_3.setScene(scene)
            self.graphicsView_3.fitInView(scene.sceneRect())
            self.repaint_widget(self.graphicsView_3)
            self.update_levels_vbf()
            self.update_line(self.statusedit, "Succesfully created VBF.")
            yshap, xshap = self.vbf_data.shape
            self.update_line(self.lineEdit_10, f"Size: {xshap}x{yshap}.")
            f.close()
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def write_to_blo(self):
        """Export the file to blo in an incremental way"""
        path_hdf5 = self.lineEdit_4.text()
        path_blo = self.lineEdit_7.text()
        try:
            # check if an hdf5 file is selected
            if not path_hdf5:
                raise Exception("No valid HDF5 file selected!")
            if not path_blo:
                raise Exception("No valid blo file selected!")
            # try to read the info from the file
            f = rec.hdf5Intermediate(path_hdf5)
            # none or 0 means default
            start_frame = None
            end_frame = None
            sdimx = None
            sdimy = None
            hyst = 0
            snakescan = True
            # overwrite standard info depending on gui
            if self.checkBox_2.checkState():
                # use custom scanning
                sdimx = self.spinBox.value()
                sdimy = self.spinBox_2.value()
            if self.checkBox_11.checkState():
                # use custom frames
                start_frame = self.spinBox_15.value()
                end_frame = self.spinBox_16.value()
            # use hysteresis or not
            if self.checkBox_6.checkState():
                hyst = self.spinBox_9.value()
            # use snake scan or not
            if self.checkBox_12.checkState():
                snakescan = True
            else:
                snakescan = False
            # calculate the image
            logger.debug(f"We try to create a blo file with data: "
                         f"S.F. {start_frame}, E.F. {end_frame}, "
                         f"Dims: x {sdimx} y {sdimy},"
                         f"hyst: {hyst}, snakescan: {snakescan}")
            logger.debug("Calculating shape and indexes")
            shape, indexes = f.get_blo_export_data(sdimx, sdimy,
                                                   start_frame,
                                                   end_frame, hyst, snakescan)
            logger.debug(f"Shape: {shape}")
            logger.debug("Starting to write blo file")
            self.update_line(self.statusedit, "Writing blo file...")
            self.get_thread = blf.bloFileWriter(f, path_blo, shape, indexes)
            self.get_thread.increase_progress.connect(self.increase_progbar)
            self.get_thread.finish.connect(self.done_bloexport)
            self.get_thread.start()
            self.window.setEnabled(False)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done_bloexport(self):
        self.window.setEnabled(True)
        # also update lines in the second pannel
        self.update_line(self.statusedit,
                         "Succesfully exported to blo")

    def export_tiffs(self):
        path_hdf5 = self.lineEdit_4.text()
        path_tiff = self.lineEdit_9.text()
        try:
            # check if an hdf5 file is selected
            if not path_hdf5:
                raise Exception("No valid HDF5 file selected!")
            if not path_tiff:
                raise Exception("No valid tiff file path selected!")
            pre, fin = os.path.splitext(path_tiff)

            f = rec.hdf5Intermediate(path_hdf5)
            tot_frames = f["Scan"].attrs["total_stream_frames"]
            first_frame = self.spinBox_5.value()
            last_frame = self.spinBox_4.value()
            dtype = self.comboBox_2.currentText()
            if dtype == "uint8":
                dtype = np.uint8
            elif dtype == "uint16":
                dtype = np.uint16
            else:
                raise Exception("Unexpected dtype")
            if tot_frames <= last_frame:
                raise Exception("Frames are out of range")
            frames = np.arange(first_frame, last_frame+1)
            self.update_line(self.statusedit, "Exporting to tiff files...")
            self.get_thread = tfe.TiffFileWriter(f, frames, dtype, pre, fin)
            self.get_thread.increase_progress.connect(self.increase_progbar)
            self.get_thread.finish.connect(self.done_tiffexport)
            self.get_thread.start()
            self.window.setEnabled(False)
        except Exception as e:
            self.update_line(self.statusedit, f"Error: {e}")

    def done_tiffexport(self):
        self.window.setEnabled(True)
        # also update lines in the second pannel
        self.update_line(self.statusedit,
                         "Succesfully exported Tiff files")

    def increase_progbar(self, value):
        self.progressBar.setValue(value)

    def hardRepaint(self):
        self.window.hide()
        self.window.show()


def main():
    app = QApplication([])
    window = Window()
    _ = ConnectedWidget(window)
    window.setWindowTitle("TVIPS converter")
    window.show()
    # form.lineEdit.setText("/Volumes/Elements/200309-2F/"
    #                       "rec_20200309_113857_000.tvips")
    # form.lineEdit_2.setText("/Users/nielscautaerts/Desktop/stream_2.hdf5")
    app.exec_()


if __name__ == "__main__":
    main()
