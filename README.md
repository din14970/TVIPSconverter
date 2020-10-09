# TVIPS converter

## Cite TVIPS converter
[![DOI](https://zenodo.org/badge/228454716.svg)](https://zenodo.org/badge/latestdoi/228454716)

## What does it do?

This tool with a Qt user interface serves to convert .tvips movie files from TVIPS CCD/CMOS cameras collected with EM-Konos or EM-Scan into more convenient formats. These kinds of cameras may be found in transmission electron microscopes (TEM) and are used for recording precession electron diffraction (PED) and 4D-STEM data, as well as in-situ experiment recordings. To analyze the data with 3rd party tools, the tvips data format is highly inconvenient; this tool aims to alleviate this issue.

Currently supported export formats:

| Export file format | Types of experiments | Analysis tools | Remarks                     |
| ------------------ | -------------------- | -------------- | --------------------------- |
| .blo               | PED/4D-STEM          | ASTAR (NanoMegas), [pyxem](http://www.pyxem.org/), [hyperspy](http://hyperspy.org/), [LiberTEM](https://libertem.github.io/LiberTEM/) | results are exported as 8-bit images, which can result in a significant loss of information if the orignal images were 16 bit|
| .hspy              | PED/4D-STEM          | [pyxem](http://www.pyxem.org/), [hyperspy](http://hyperspy.org/), [LiberTEM](https://libertem.github.io/LiberTEM/) | images are exported at full depth. ASTAR does not open this file format. A .hspy file is a specially structured HDF5 file. |
| .tiff              | any                  | Any image processing software | For exporting a subset of the images individually. |
| converter hdf5     | all                  | TVIPSconverter | An intermediate hdf5 file that contains all the TVIPS metadata and pre-processed 16-bit images. Serves to easily explore the data with [HDFView](https://www.hdfgroup.org/downloads/hdfview/). In a future version, this intermediate file may disappear or change in favor of directly making a hyperspy compatible file |

In the future we aim to support:

* USID HDF5 format

## How do I install it? (recommended method)

1. Install [Anaconda](https://www.anaconda.com/distribution/).
2. Create a new virtual environment on your system to install tvipsconverter in

	```
	$ conda create --name tvipsconverter pip
	```

3. Activate the virtual environment anywhere in your system with

	```
	$ conda activate tvipsconverter
	```

4. Pip install `tvipsconverter`

	```
	$ pip install tvipsconverter
	```

**Note:** On Windows you will have to type these commands in the `Anaconda prompt`

## How do I use it?

1. Activate the `tvipsconverter` virtual environment.
2. Run the command `$ tvipsconverter` which should bring up the GUI after some time.
3. The GUI should be self-explanatory with tool tips. However, I have also made a video briefly demonstrating the GUI. You can find it [here](https://youtu.be/ZvbQn8fq4_M).

## How does it work?

When recording data on a TVIPS camera, it is stored as a long collection of images (a movie),
separated into multiple large .tvips files. TVIPS is a small German company and their tools
are still in active development. As of version 2 of the TVIPS recording software,
very limited information on microscopy settings is stored in this file. **In fact,
the image pixel size and scan pixel dimensions are not properly recorded inside the file.
You must record these settings and correct this manually!** If the aim is to
record a 4D-STEM or PED dataset, this format is difficult to work with. Converting
the format to formats that can be read by other software is the main aim of this tool.
The tool first converts the multiple .tvips files to a single easy to manage and browse .hdf5 file.
In the second step, this hdf5 file can be converted to other types of files (see table above).
The conversion process may depend on which acquisition tool that was used to collect the
tvips data.

### EM-Konos
If the recorder software was EM-Konos, there is no direct information stored related to a scan. 
The only information available to correlate the images to the electron beam scan is
the so called `rotator` index, which is stored in the header of each image.
It corresponds to the index of the scan point to which the image should be mapped.
The problem is that the electron beam dwell time and the frame acquisition time
may not be perfectly aligned, with the result that some `rotator` images are repeated
while some scan points have no corresponding image. In addition, the scanning control
follows a snake-like meander pattern instead of a line-by-line fly-back.
This causes a hysteresis artefact (offset) on the even scan lines.
TVIPSconverter provides options to correct for these artefacts and as best as possible
align the images to the scan. With EM-Konos data, an automatic guess is offered, but
the user can also manually specify start and stop frames.

### EM-Scan
EM-Scan is the second iteration of acquisition software and does store scan info in the
image headers. There is no more rotator index and scanning follows a regular fly-back pattern.
At present, TVIPSconverter does not make use of this metadata. The software can be a bit
buggy and the scan positions can be absent anyway. With EM-Scan data, the user will have to
supply the scan information manually.

## Credits and notes

**This tool is not an official product of the TVIPS company. Use at your own risk. 
I am not responsbile for loss or corruption of data.** The tool derives from python scripts
originally developed by the company. We have significantly modified these
scripts mainly to make the conversion process possible on a computer with regular sized RAM and
support loss-less export to hdf5. The GUI is also our addition.

## Changelog

### 0.1.0

* It is now possible to export to Pyxem (Hyperspy) hdf5. The depth of the images is conserved.
* The user can now set the scan and diffraction pattern resolution in the GUI.
* The GUI was slightly simplified and some tooltips were updated.
* The user can now select which frame to visualize when performing the pre-processing.

### 0.0.8

* fixed image rebin problem

### 0.0.7

* added support for EM-Scan data where scanning doesn't follow snake pattern
* fixed some bugs
