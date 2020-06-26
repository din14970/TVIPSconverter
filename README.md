# TVIPS converter

## What does it do?

This tool with a Qt user interface serves to convert .tvips movie files from TVIPS CCD/CMOS cameras into more convenient formats. These kinds of cameras may be found in transmission electron microscopes (TEM) and are used for recording precession electron diffraction (PED) and 4D-STEM data, as well as in-situ experiment recordings. To analyze the data with 3rd party tools, the tvips data format is highly inconvenient; this tool aims to alleviate this issue.

Currently supported export formats:

* .blo (for PED) - can be analyzed with [pyxem](http://www.pyxem.org/) or NanoMegas software
* .tiff

In the future we aim to support:

* Hyperspy HDF5 format
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

When recording data on a TVIPS camera, it is stored as a long collection of images,
separated into multiple large .tvips files. TVIPS is a small German company and their tools
are still in active development. As of version 2 of the TVIPS recording software,
very limited information on microscopy settings is stored in this file. **In fact,
the image pixel size and scan pixel dimensions are not properly recorded inside the file.
You must record these settings and correct this manually in the output files later with other tools!**
There is also no direct information stored related to a scan. If the aim is to
record is a 4D-STEM or PED dataset, this format is difficult to work with.
The only information available to correlate the images to the electron beam scan is
the so called `rotator` index, which is stored in the header of each image.
It corresponds to the index of the scan point to which the image should be mapped.
The problem is that the electron beam dwell time and the frame acquisition time
may not be perfectly aligned, with the result that some `rotator` images are repeated
while some scan points have no corresponding image. In addition, the scanning control
of the TVIPS software is unusual: they follow a snake-like meander pattern instead of
a line-by-line fly-back. This causes a hysteresis artefact (offset) on the even scan lines.
This software aims to provide all the tools to correct these artefacts as best as possible and
convert scan data into formats that is directly interpretable by other software. To do so
we first convert the multiple .tvips files to a single easy to manage and browse .hdf5 file.
This is so that not all data must remain in RAM. In the second step, this hdf5 file can be converted
to other types of files.

## Credits and notes

This tool is not an official product of the TVIPS company. However it is built on some scripts
originally developed by the company. We have significantly modified these
scripts mainly to make the conversion process possible on a computer with regular sized RAM by
working with an intermediate HDF5 file. The GUI is also our addition.

## Changelog

### 0.0.7

* added support for EM-Scan data where scanning doesn't follow snake pattern