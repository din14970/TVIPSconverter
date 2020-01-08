from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem,
QErrorMessage, QMessageBox, QPlainTextEdit, QMainWindow)
from PyQt5.QtGui import QImage, QPixmap
from time import sleep
from utils import recorder
import logging
import os

rawgui, Window = uic.loadUiType("widget_2.ui")

class ConnectedWidget(rawgui):

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setupUi(window)
        self.connectUI()

    def connectUI(self):
        #initial browse button
        self.pushButton.clicked.connect(
            lambda: self.openFileBrowser(self.lineEdit, "TVIPS (*.tvips);;\
                                                        BLO (*.blo);;\
                                                        TIFF (*.tiff *.tif)"))
        #select a folder
        self.pushButton_2.clicked.connect(
            lambda: self.openFolderBrowser(self.lineEdit_2))

        #update button on the preview
        self.pushButton_4.clicked.connect(self.updatePreviewDebug)

        #execute the conversion command
        self.pushButton_3.clicked.connect(self.exportFiles)

        #deactivate part of the gui upon activation
        self.checkBox_8.stateChanged.connect(self.updateActive)
        self.checkBox_3.stateChanged.connect(self.updateActive)
        self.checkBox_7.stateChanged.connect(self.updateActive)
        self.checkBox_5.stateChanged.connect(self.updateActive)
        self.checkBox_4.stateChanged.connect(self.updateActive)
        self.checkBox_6.stateChanged.connect(self.updateActive)
        self.updateActive()


    def updateActive(self):
        if self.checkBox_8.checkState()==2:
            self.spinBox_11.setEnabled(True)
        else:
            self.spinBox_11.setEnabled(False)

        if self.checkBox_3.checkState()==2:
            self.spinBox_6.setEnabled(True)
            self.doubleSpinBox.setEnabled(True)
        else:
            self.spinBox_6.setEnabled(False)
            self.doubleSpinBox.setEnabled(False)

        if self.checkBox_7.checkState()==2:
            self.spinBox_10.setEnabled(True)
        else:
            self.spinBox_10.setEnabled(False)

        if self.checkBox_5.checkState()==2:
            self.spinBox_3.setEnabled(True)
        else:
            self.spinBox_3.setEnabled(False)

        if self.checkBox_4.checkState()==2:
            self.spinBox_7.setEnabled(True)
            self.spinBox_8.setEnabled(True)
        else:
            self.spinBox_7.setEnabled(False)
            self.spinBox_8.setEnabled(False)

        if self.checkBox_6.checkState()==2:
            self.spinBox_9.setEnabled(True)
        else:
            self.spinBox_9.setEnabled(False)

    def openFileBrowser(self, le, fs):
        path, okpres = QFileDialog.getOpenFileName(caption = "Select file", filter = fs)
        if okpres:
            le.setText(path)

    def openFolderBrowser(self, le):
        path = QFileDialog.getExistingDirectory(caption = "Choose directory")
        if path:
            le.setText(path)

    def updatePreview(self):
        self.readValues()
        #create one image properly processed
        try:
            recorder.createOneImageUI(**self.uivalues)
            #plot this figure
            image = QImage('./temp.tiff')
            pixmap = QPixmap.fromImage(image)
            scene = QGraphicsScene()
            scene.addItem(QGraphicsPixmapItem(pixmap))
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect())
            self.statusedit.setText("Status: Succesfully created preview.")
            shpx = image.width()
            shpy = image.height()
            self.lineEdit_4.setText("Image size: {}x{} pixels".format(shpx, shpy))
            self.hardRepaint()
        except:
            self.statusedit.setText("Status: An error occured while creating preview, please double-check setting!")
            self.hardRepaint()
            # b= QErrorMessage()
            # b.showMessage("Invalid arguments! Check file path.")
            # b.exec_()

    def updatePreviewDebug(self):
        self.readValues()
        #create one image properly processed
        recorder.createOneImageUI(**self.uivalues)
        #plot this figure
        image = QImage('./temp.tiff')
        pixmap = QPixmap.fromImage(image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.sceneRect())
        self.statusedit.setText("Status: Succesfully created preview.")
        shpx = image.width()
        shpy = image.height()
        self.lineEdit_4.setText("Image size: {}x{} pixels".format(shpx, shpy))
        self.hardRepaint()


    def exportFiles(self):
        self.readValues()
        # #create one image properly processed
        # k = QMainWindow()
        # log_handler = QPlainTextEditLogger(k)
        # logging.getLogger().addHandler(log_handler)
        # k.show()

        condition =(os.path.exists(self.uivalues["inp"]) and
                    os.path.exists(self.uivalues["oup"]) and
                    self.uivalues["pref"]!="")

        if condition:
            self.statusedit.setText("Status: Busy ...")
            self.hardRepaint()

            recorder.mainUI(**self.uivalues)
            self.statusedit.setText("Status: Done converting file.")
            self.hardRepaint()
            # try:
            #     recorder.mainUI(**self.uivalues)
            #     self.statusedit.setText("Status: Done converting file.")
            #     self.hardRepaint()
            # except:
            #     self.statusedit.setText("Status: An error occured while converting. Double-check setting!")
            #     self.hardRepaint()
                # b= QErrorMessage()
                # b.showMessage("Invalid arguments! Check file path.")
                # b.exec_()
        else:
            self.statusedit.setText("Status: Must provide valid input and output arguments.")
            self.hardRepaint()

    def readValues(self):
        '''Read and return the current UI values'''
        self.uivalues = {
        "inp": self.lineEdit.text(), #input path
        "oup": self.lineEdit_2.text(), #output path
        "pref": self.lineEdit_3.text(), #prefix out
        "oupt": self.comboBox.currentText(), #output type
        "dep": self.comboBox_2.currentText(), #output depth (uint8, uint16, int16)
        "sdx": self.spinBox.value(), #scanning dimension x
        "sdy": self.spinBox_2.value(), #scanning dimension y
        "use_bin": self.checkBox_5.checkState(), #use binning
        "bin_fac": self.spinBox_3.value(), #binning factor
        "use_scaling": self.checkBox_4.checkState(), #use scaling
        "scalemin": self.spinBox_7.value(), #scaling minimum intensiy
        "scalemax": self.spinBox_8.value(), #scaling maximum intensity
        "use_gauss": self.checkBox_3.checkState(), #use gaussian filter
        "gauss_ks": self.spinBox_6.value(), #gaussian kernel size
        "gauss_sig": self.doubleSpinBox.value(), #gaussian sigma
        "use_med": self.checkBox_7.checkState(),#use median filter
        "med_ks": self.spinBox_10.value(), #median filter kernel size
        "use_rotator": self.checkBox.checkState(), #use rotator
        "use_hyst": self.checkBox_6.checkState(), #use hysteresis
        "hyst_val": self.spinBox_9.value(), #value of hysteresis
        "skip": self.spinBox_4.value(), #skipp first frames
        "trunc": self.spinBox_5.value(), #skipp last frames
        "useintcut": self.checkBox_8.checkState(),
        "intcut":self.spinBox_11.value()
        }


    def hardRepaint(self):
        self.window.hide()
        self.window.show()

# class QPlainTextEditLogger(logging.Handler):
#     def __init__(self, parent):
#         super().__init__()
#
#         self.widget = QPlainTextEdit(parent)
#         self.widget.setReadOnly(True)
#
#     def emit(self, record):
#         msg = self.format(record)
#         self.widget.textCursor().appendPlainText(msg)
#
#     def write(self, m):
#         pass

def main():
    app = QApplication([])
    window = Window()
    form = ConnectedWidget(window)
    window.setWindowTitle("TVIPS / blo converter")
    window.show()
    app.exec_()

def mainDebug():
    app = QApplication([])
    window = Window()
    form = ConnectedWidget(window)
    window.setWindowTitle("TVIPS / blo converter")
    window.show()
    #set the values
    form.lineEdit.setText("./Dummy/rec_20190412_183600_000.tvips")
    form.spinBox.setValue(150)
    form.spinBox_2.setValue(1)
    form.lineEdit_2.setText("./Dummy")
    form.lineEdit_3.setText("testpref")
    form.comboBox.setCurrentIndex(1)
    app.exec_()

if __name__ == "__main__":
    mainDebug()
