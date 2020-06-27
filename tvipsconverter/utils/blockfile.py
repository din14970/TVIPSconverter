# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import logging
import warnings
import datetime
import dateutil

from dateutil import tz, parser

from . import imagefun

from collections import OrderedDict

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


def sarray2dict(sarray, dictionary=None):
    """Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dict
        If dictionary is not None the content of sarray will be appended to the
        given dictonary

    Returns
    -------
    Ordered dictionary

    """
    if dictionary is None:
        dictionary = OrderedDict()
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 \
            else sarray[name]
    return dictionary


def dict2sarray(dictionary, sarray=None, dtype=None):
    """Populates a struct array from a dictionary

    Parameters
    ----------
    dictionary: dict
    sarray: struct array or None
        Either sarray or dtype must be given. If sarray is given, it is
        populated from the dictionary.
    dtype: None, numpy dtype or dtype list
        If sarray is None, dtype must be given. If so, a new struct array
        is created according to the dtype, which is then populated.

    Returns
    -------
    Structure array

    """
    if sarray is None:
        if dtype is None:
            raise ValueError("Either sarray or dtype need to be specified.")
        sarray = np.zeros((1,), dtype=dtype)
    for name in set(sarray.dtype.names).intersection(set(dictionary.keys())):
        if len(sarray[name]) == 1:
            sarray[name][0] = dictionary[name]
        else:
            sarray[name] = dictionary[name]
    return sarray


def ISO_format_to_serial_date(date, time, timezone='UTC'):
    """ Convert ISO format to a serial date. """
    if timezone is None or timezone == 'Coordinated Universal Time':
        timezone = 'UTC'
    dt = parser.parse(
        '%sT%s' %
        (date, time)).replace(
        tzinfo=tz.gettz(timezone))
    return datetime_to_serial_date(dt)


def datetime_to_serial_date(dt):
    """ Convert datetime.datetime object to a serial date. """
    if dt.tzname() is None:
        dt = dt.replace(tzinfo=tz.tzutc())
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    delta = dt - origin
    return float(delta.days) + (float(delta.seconds) / 86400.0)


def serial_date_to_datetime(serial):
    """ Convert serial date to a datetime.datetime object. """
    # Excel date&time format
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    secs = (serial % 1.0) * 86400
    delta = datetime.timedelta(int(serial), secs)
    return origin + delta


def serial_date_to_ISO_format(serial):
    """
    Convert serial_date to a tuple of string (date, time, time_zone) in ISO
    format. By default, the serial date is converted in local time zone.
    """
    dt_utc = serial_date_to_datetime(serial)
    dt_local = dt_utc.astimezone(tz.tzlocal())
    return (dt_local.date().isoformat(), dt_local.time().isoformat(),
            dt_local.tzname())


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'Blockfile'
description = 'Read/write support for ASTAR blockfiles'
full_support = False
# Recognised file extension
file_extensions = ['blo', 'BLO']
default_extension = 0

# Writing capabilities:
writes = [(2, 2), (2, 1), (2, 0)]
magics = [0x0102]


mapping = {
    'blockfile_header.Beam_energy':
    ("Acquisition_instrument.TEM.beam_energy", lambda x: x * 1e-3),
    'blockfile_header.Camera_length':
    ("Acquisition_instrument.TEM.camera_length", lambda x: x * 1e-4),
    'blockfile_header.Scan_rotation':
    ("Acquisition_instrument.TEM.rotation", lambda x: x * 1e-2),
}


def get_header_dtype_list(endianess='<'):
    end = endianess
    dtype_list = \
        [
            ('ID', (bytes, 6)),
            ('MAGIC', end + 'u2'),
            ('Data_offset_1', end + 'u4'),      # Offset VBF
            ('Data_offset_2', end + 'u4'),      # Offset DPs
            ('UNKNOWN1', end + 'u4'),           # Flags for ASTAR software?
            ('DP_SZ', end + 'u2'),              # Pixel dim DPs
            ('DP_rotation', end + 'u2'),        # [degrees ( * 100 ?)]
            ('NX', end + 'u2'),                 # Scan dim 1
            ('NY', end + 'u2'),                 # Scan dim 2
            ('Scan_rotation', end + 'u2'),      # [100 * degrees]
            ('SX', end + 'f8'),                 # Pixel size [nm]
            ('SY', end + 'f8'),                 # Pixel size [nm]
            ('Beam_energy', end + 'u4'),        # [V]
            ('SDP', end + 'u2'),                # Pixel size [100 * ppcm]
            ('Camera_length', end + 'u4'),      # [10 * mm]
            ('Acquisition_time', end + 'f8'),   # [Serial date]
        ] + [
            ('Centering_N%d' % i, 'f8') for i in range(8)
        ] + [
            ('Distortion_N%02d' % i, 'f8') for i in range(14)
        ]

    return dtype_list


def get_default_header(endianess='<'):
    """Returns a header pre-populated with default values.
    """
    dt = np.dtype(get_header_dtype_list())
    header = np.zeros((1,), dtype=dt)
    header['ID'][0] = 'IMGBLO'.encode()
    header['MAGIC'][0] = magics[0]
    header['Data_offset_1'][0] = 0x1000     # Always this value observed
    header['UNKNOWN1'][0] = 131141          # Very typical value (always?)
    header['Acquisition_time'][0] = datetime_to_serial_date(
        datetime.datetime.fromtimestamp(86400, dateutil.tz.tzutc()))
    return header


def get_header_from_signal(signal, endianess='<'):
    header = get_default_header(endianess)
    if 'blockfile_header' in signal.original_metadata:
        header = dict2sarray(signal.original_metadata['blockfile_header'],
                             sarray=header)
        note = signal.original_metadata['blockfile_header']['Note']
    else:
        note = ''
    if signal.axes_manager.navigation_dimension == 2:
        NX, NY = signal.axes_manager.navigation_shape
        SX = signal.axes_manager.navigation_axes[0].scale
        SY = signal.axes_manager.navigation_axes[1].scale
    elif signal.axes_manager.navigation_dimension == 1:
        NX = signal.axes_manager.navigation_shape[0]
        NY = 1
        SX = signal.axes_manager.navigation_axes[0].scale
        SY = SX
    elif signal.axes_manager.navigation_dimension == 0:
        NX = NY = SX = SY = 1

    DP_SZ = signal.axes_manager.signal_shape
    if DP_SZ[0] != DP_SZ[1]:
        raise ValueError('Blockfiles require signal shape to be square!')
    DP_SZ = DP_SZ[0]
    SDP = 100. / signal.axes_manager.signal_axes[0].scale

    offset2 = NX * NY + header['Data_offset_1']
    # Based on inspected files, the DPs are stored at 16-bit boundary...
    # Normally, you'd expect word alignment (32-bits) ¯\_(°_o)_/¯
    offset2 += offset2 % 16

    header_sofar = {
        'NX': NX, 'NY': NY,
        'DP_SZ': DP_SZ,
        'SX': SX, 'SY': SY,
        'SDP': SDP,
        'Data_offset_2': offset2,
    }

    header = dict2sarray(header_sofar, sarray=header)
    return header, note


def get_header(data_shape, scan_scale, diff_scale, endianess="<", **kwargs):
    header = get_default_header(endianess)
    note = ''
    if len(data_shape) == 4:
        NY, NX = data_shape[:2][::-1]  # first dimension seems to by y in np
        SX = scan_scale
        SY = scan_scale
    elif len(data_shape) == 3:
        NX = data_shape[0]
        NY = 1
        SX = scan_scale
        SY = SX
    elif len(data_shape) == 2:
        NX = NY = SX = SY = 1
    else:
        raise ValueError("Invalid data shape")

    DP_SZ = data_shape[-2:]
    if DP_SZ[0] != DP_SZ[1]:
        raise ValueError('Blockfiles require signal shape to be square!')
    DP_SZ = DP_SZ[0]
    SDP = 100. / diff_scale

    offset2 = NX * NY + header['Data_offset_1']
    # Based on inspected files, the DPs are stored at 16-bit boundary...
    # Normally, you'd expect word alignment (32-bits) ¯\_(°_o)_/¯
    offset2 += offset2 % 16

    header_sofar = {
        'NX': NX, 'NY': NY,
        'DP_SZ': DP_SZ,
        'SX': SX, 'SY': SY,
        'SDP': SDP,
        'Data_offset_2': offset2,
    }

    header_sofar.update(kwargs)
    header = dict2sarray(header_sofar, sarray=header)

    return header, note


def file_reader(filename, endianess='<', mmap_mode=None,
                lazy=False, **kwds):
    _logger.debug("Reading blockfile: %s" % filename)
    metadata = {}
    if mmap_mode is None:
        mmap_mode = 'r' if lazy else 'c'
    # Makes sure we open in right mode:
    if '+' in mmap_mode or ('write' in mmap_mode and
                            'copyonwrite' != mmap_mode):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, 'r+b')
    else:
        f = open(filename, 'rb')
    _logger.debug("File opened")

    # Get header
    header = np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
    if header['MAGIC'][0] not in magics:
        warnings.warn("Blockfile has unrecognized header signature. "
                      "Will attempt to read, but correcteness not guaranteed!")
    header = sarray2dict(header)
    note = f.read(header['Data_offset_1'] - f.tell())
    # It seems it uses "\x00" for padding, so we remove it
    try:
        header['Note'] = note.decode("latin1").strip("\x00")
    except Exception:
        # Not sure about the encoding so, if it fails, we carry on
        _logger.warn(
            "Reading the Note metadata of this file failed. "
            "You can help improving "
            "HyperSpy by reporting the issue in "
            "https://github.com/hyperspy/hyperspy")
    _logger.debug("File header: " + str(header))
    NX, NY = int(header['NX']), int(header['NY'])
    DP_SZ = int(header['DP_SZ'])
    if header['SDP']:
        SDP = 100. / header['SDP']
    else:
        SDP = -1
    original_metadata = {'blockfile_header': header}

    # Get data:

    # A Virtual BF/DF is stored first
    offset1 = header['Data_offset_1']
    f.seek(offset1)
    data_pre = np.fromfile(f, count=NX*NY, dtype=endianess+'u1'
                           ).squeeze().reshape((NY, NX), order='C')

    # Then comes actual blockfile
    offset2 = header['Data_offset_2']
    if not lazy:
        f.seek(offset2)
        data = np.fromfile(f, dtype=endianess + 'u1')
    else:
        data = np.memmap(f, mode=mmap_mode, offset=offset2,
                         dtype=endianess + 'u1')
    try:
        data = data.reshape((NY, NX, DP_SZ * DP_SZ + 6))
    except ValueError:
        warnings.warn(
            'Blockfile header dimensions larger than file size! '
            'Will attempt to load by zero padding incomplete frames.')
        # Data is stored DP by DP:
        pw = [(0, NX * NY * (DP_SZ * DP_SZ + 6) - data.size)]
        data = np.pad(data, pw, mode='constant')
        data = data.reshape((NY, NX, DP_SZ * DP_SZ + 6))

    # Every frame is preceeded by a 6 byte sequence (AA 55, and then a 4 byte
    # integer specifying frame number)
    data = data[:, :, 6:]
    data = data.reshape((NY, NX, DP_SZ, DP_SZ), order='C').squeeze()

    units = ['nm', 'nm', 'cm', 'cm']
    names = ['y', 'x', 'dy', 'dx']
    scales = [header['SY'], header['SX'], SDP, SDP]
    date, time, time_zone = serial_date_to_ISO_format(
        header['Acquisition_time'])
    metadata = {'General': {'original_filename': os.path.split(filename)[1],
                            'date': date,
                            'time': time,
                            'time_zone': time_zone,
                            'notes': header['Note']},
                "Signal": {'signal_type': "diffraction",
                           'record_by': 'image', },
                }
    # Create the axis objects for each axis
    dim = data.ndim
    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': 0.0,
            'units': units[i], }
        for i in range(dim)]

    dictionary = {'data': data,
                  'vbf': data_pre,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': original_metadata,
                  'mapping': mapping, }

    f.close()
    return [dictionary, ]


def file_writer(filename, signal, **kwds):
    endianess = kwds.pop('endianess', '<')
    header, note = get_header_from_signal(signal, endianess=endianess)

    with open(filename, 'wb') as f:
        # Write header
        header.tofile(f)
        # Write header note field:
        if len(note) > int(header['Data_offset_1']) - f.tell():
            note = note[:int(header['Data_offset_1']) - f.tell() - len(note)]
        f.write(note.encode())
        # Zero pad until next data block
        zero_pad = int(header['Data_offset_1']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)
        # Write virtual bright field
        vbf = signal.mean(
            signal.axes_manager.signal_axes[
                :2]).data.astype(
            endianess +
            'u1')
        vbf.tofile(f)
        # Zero pad until next data block

        if f.tell() > int(header['Data_offset_2']):
            raise ValueError("Signal navigation size does not match "
                             "data dimensions.")
        zero_pad = int(header['Data_offset_2']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)

        # Write full data stack:
        # We need to pad each image with magic 'AA55', then a u32 serial
        dp_head = np.zeros((1,), dtype=[('MAGIC', endianess + 'u2'),
                                        ('ID', endianess + 'u4')])
        dp_head['MAGIC'] = 0x55AA
        # Write by loop:
        for img in signal._iterate_signal():
            dp_head.tofile(f)
            img.astype(endianess + 'u1').tofile(f)
            dp_head['ID'] += 1


class bloFileWriter(QThread):
    """Write a blo file from an HDF5 without loading all data in memory"""
    increase_progress = pyqtSignal(int)
    finish = pyqtSignal()

    def __init__(self, fh, path_blo, shape, indexes):
        QThread.__init__(self)
        self.fh = fh  # open hdf5 file in read mode
        self.path_blo = path_blo
        self.shape = shape  # shape of hypercube
        self.indexes = indexes  # indexes: right order of frames
        # scale seems unknown and arbitrarily determined in original code
        self.scan_scale = 5  # TODO can this be extracted from the TVIPS meta?
        self.diff_scale = 1.075  # TODO can this be extracted from TVIPS meta?
        vbfs = self.fh["Scan"]["vbf_intensities"][:]
        self.vbf_im = vbfs[self.indexes].reshape(self.shape[0], self.shape[1])
        self.vbf_im = imagefun.normalize_convert(self.vbf_im, dtype=np.uint8)
        logger.debug("Initialized bloFileWriter")

    def run(self):
        self.convert_to_blo()
        self.finish.emit()
        self.fh.close()  # close the hdf5 file

    def convert_to_blo(self):
        endianess = "<"
        header, note = get_header(
            self.shape, self.scan_scale,
            self.diff_scale, endianess,
            Camera_length=100.0*self.fh["ImageStream"].attrs['magtotal'],
            Beam_energy=self.fh["ImageStream"].attrs['ht']*1000,
            Distortion_N01=1.0, Distortion_N09=1.0,
            Note="Reconstructed from TVIPS image stream")

        logger.debug("Created header of blo file")
        with open(self.path_blo, "wb") as f:
            # Write header
            header.tofile(f)
            logger.debug("Wrote header to file")
            # Write header note field:
            if len(note) > int(header['Data_offset_1']) - f.tell():
                note = note[:int(header['Data_offset_1']) -
                            f.tell() - len(note)]
            f.write(note.encode())
            # Zero pad until next data block
            zero_pad = int(header['Data_offset_1']) - f.tell()
            np.zeros((zero_pad,), np.byte).tofile(f)
            # Write virtual bright field
            vbf = self.vbf_im.astype(endianess + "u1")
            vbf.tofile(f)
            # Zero pad until next data block
            if f.tell() > int(header['Data_offset_2']):
                raise ValueError("Signal navigation size does not match "
                                 "data dimensions.")
            zero_pad = int(header['Data_offset_2']) - f.tell()
            np.zeros((zero_pad,), np.byte).tofile(f)
            # Write full data stack:
            # We need to pad each image with magic 'AA55', then a u32 serial
            dp_head = np.zeros((1,), dtype=[('MAGIC', endianess + 'u2'),
                                            ('ID', endianess + 'u4')])
            dp_head['MAGIC'] = 0x55AA
            # Write by loop:
            logger.debug("Wrote header part of blo file")
            for j, indx in enumerate(self.indexes):
                dp_head.tofile(f)
                c = f"{indx}".zfill(6)
                img = self.fh["ImageStream"][f"Frame_{c}"][:]
                img = imagefun.normalize_convert(img, dtype=np.uint8)
                img.astype(endianess + 'u1').tofile(f)
                dp_head['ID'] += 1
                self.update_gui_progress(j)
                logger.debug(f"Wrote frame Frame_{c} to blo file")

    def update_gui_progress(self, j):
        """If using the GUI update features with progress"""
        value = int((j+1)/len(self.indexes)*100)
        self.increase_progress.emit(value)


def file_writer_array(filename, array, scan_scale, diff_scale, **kwds):
    endianess = "<"
    header, note = get_header(array.shape, scan_scale, diff_scale, endianess,
                              **kwds)

    with open(filename, 'wb') as f:
        # Write header
        header.tofile(f)
        # Write header note field:
        if len(note) > int(header['Data_offset_1']) - f.tell():
            note = note[:int(header['Data_offset_1']) - f.tell() - len(note)]
        f.write(note.encode())
        # Zero pad until next data block
        zero_pad = int(header['Data_offset_1']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)
        # Write virtual bright field
        vbf = None

        if len(array.shape) == 4:
            amean = (2, 3)
            # do proper vbf
            xx, yy = np.meshgrid(range(array.shape[2]), range(array.shape[3]))
            mask = np.hypot(xx - 0.5 * array.shape[2],
                            yy - 0.5 * array.shape[3]) < 5
            # TODO: make radius and offset configurable

            vbffloat = np.zeros((array.shape[:2]))
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    vbffloat[i, j] = array[i, j][mask].sum()

            # scale to 8 bit
            vbf = imagefun.scalestd(vbffloat).astype(endianess + "u1")

        elif len(array.shape) == 3:
            amean = (1, 2)

            vbf = array.mean(axis=amean).astype(endianess + 'u1')
        vbf.tofile(f)
        # Zero pad until next data block
        if f.tell() > int(header['Data_offset_2']):
            raise ValueError("Signal navigation size does not match "
                             "data dimensions.")
        zero_pad = int(header['Data_offset_2']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)

        # Write full data stack:
        # We need to pad each image with magic 'AA55', then a u32 serial
        dp_head = np.zeros((1,), dtype=[('MAGIC', endianess + 'u2'),
                                        ('ID', endianess + 'u4')])
        dp_head['MAGIC'] = 0x55AA
        # Write by loop:
        for img in array.reshape(array.shape[0]*array.shape[1],
                                 *array.shape[2:]):
            dp_head.tofile(f)
            img.astype(endianess + 'u1').tofile(f)
            dp_head['ID'] += 1
