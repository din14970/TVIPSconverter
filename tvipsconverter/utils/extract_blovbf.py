from . import blockfile
import argparse
import tifffile

def main():
    parser = argparse.ArgumentParser(description='Process .tvips recorder format')

    parser.add_argument('input', help="Input filename, must be .blo")
    parser.add_argument('output', help="Output filename")

    opts = parser.parse_args()

    assert (opts.input.endswith(".blo"))

    blofile = blockfile.file_reader(opts.input)

    tifffile.imsave(opts.output, blofile[0]['vbf'])

if __name__ == "__main__":
    main()
