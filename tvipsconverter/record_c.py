"""
Converts the TVIPS to a blo file from the command line
"""

import re
import sys

from utils.recorder import mainCLI

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(mainCLI())
