import tifffile
from tvips.utils import em
from tifffile import FileHandle

class TvipsCameraFormat(object):
    """All the variables of a camera format"""

    group = 0
    readoutmode = 0
    gainindex=0
    speedindex=0
    dimensionx=0
    dimensiony=0
    binningx=1
    binningy=1
    offsetx=0
    offsety=0

    exposuretime=0

    def __init__(self, cameragroup=0, readoutmode=0, gainindex=0, speedindex=0, dimensionx=0, dimensiony=0, binningx=1, binningy=1, offsetx=0, offsety=0, exposuretime=0):
        self.group = cameragroup
        self.readoutmode = readoutmode
        self.gainindex = gainindex
        self.speedindex = speedindex
        self.dimensionx = dimensionx
        self.dimensiony = dimensiony
        self.binningx = binningx
        self.binningy = binningy
        self.offsetx = offsetx
        self.offsety = offsety
        self.exposuretime = exposuretime

    @staticmethod
    def from_tvips_tiff(path):
        with tifffile.TiffFile as tif:
            assert tif.is_tvips
            return TvipsCameraFormat.from_tvips_tiff_header(tif.pages[0].tvips_metadata)


    @staticmethod
    def from_tvips_tiff_header(hdr):
            return TvipsCameraFormat(


            int(hdr.cam_misc[0]), #group
            int(hdr.cam_misc[8]), #readout_mode 1=normal, 2=FT, 3=RS
            int(hdr.cam_misc[1]), #gain
            int(hdr.cam_misc[2]), #speed
            int(hdr.image_size_x), #dimension
            int(hdr.image_size_y), #dimension
            int(hdr.binning_x), #binning
            int(hdr.binning_y), #binning
            int(hdr.offset_x), #offset
            int(hdr.offset_y), #
            int(hdr.exposure_time)
         )

    @staticmethod
    def from_tvips_em_header(hdr):
        return TvipsCameraFormat(
            hdr.ffcameragroup,
            hdr.ffcameramode,
            hdr.ffgainindex,
            hdr.ffspeedindex,
            hdr.ffdimx,
            hdr.ffdimy,
            hdr.ffbinx,
            hdr.ffbiny,
            hdr.ffoffsetx,
            hdr.ffoffsety,
            hdr.ffexposuretime
            )

    @staticmethod
    def from_tvips_em(path):
        #shortcut: do not read entire file, just the header!
        with FileHandle(path) as f:
            hdr = f.read_record(em.EM_FORMAT_HEADER)
            return TvipsCameraFormat.from_tvips_em_header(hdr)


    def __eq__(self, other):
        #exposuretime not relevant
        assert isinstance(other, TvipsCameraFormat)

        res = self.group == other.group and\
            self.gainindex == other.gainindex and\
            self.speedindex == other.speedindex and\
            self.dimensionx == other.dimensionx and\
            self.dimensiony == other.dimensiony and\
            self.binningx == other.binningx and\
            self.binningy == other.binningy and\
            self.offsetx == other.offsetx and\
            self.offsety == other.offsety

        # the XF is only run in RS mode
        if self.group != 14:
            res = res and self.readoutmode == other.readoutmode

        return res

    def __str__(self):
        return "Group:{} Mode:{} Gain:{} Speed:{}\n DimXY:{}|{}\nBinXY:{}|{}\nOffsetXY:{}|{}".format(
            self.group,
            self.readoutmode,
            self.gainindex,
            self.speedindex,
            self.dimensionx, self.dimensiony,
            self.binningx, self.binningy,
            self.offsetx, self.offsety)

    def __hash__(self):
        #exposuretime not relevant

        #readoutmode irrelevant for XF
        if self.group != 14:
            res = hash(
                (
                self.group,
                self.readoutmode,
                self.gainindex,
                self.speedindex,
                self.dimensionx, self.dimensiony,
                self.binningx, self.binningy,
                self.offsetx, self.offsety
                )
            )
        else:
            res = hash(
                (
                self.group,
                self.gainindex,
                self.speedindex,
                self.dimensionx, self.dimensiony,
                self.binningx, self.binningy,
                self.offsetx, self.offsety
                )
            )

        return res
