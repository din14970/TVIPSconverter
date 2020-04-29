import sys
import logging
from types import SimpleNamespace
sys.path.append(".")
from tvipsconverter.utils import recorder

path_to_file = "/Volumes/Elements/200309-2F/rec_20200309_162346_000.tvips"

logging.basicConfig(level=logging.DEBUG)


def test_to_hdf5():
    """Read the first few frames"""
    d = {}
    d["numframes"] = 2000
    d["input"] = path_to_file
    d["output"] = "/Users/nielscautaerts/Desktop/"
    opts = SimpleNamespace(**d)
    recorder.Recorder(opts)


if __name__ == "__main__":
    test_to_hdf5()
