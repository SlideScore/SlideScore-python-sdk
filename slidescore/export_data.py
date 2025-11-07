DESC = """
This program converts the Anno2 binned file to a common data format, like TSV, GeoJSON, or PNG. 
Author: Bart.
"""

import sys
import argparse
import json
import gzip
import zipfile

from .lib.Decoder import Decoder
from .lib.utils import get_logger

def main(argv=None):
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('--anno2-path', '-i', type=str, required=True,
                        help='Input file path, should be an Anno2 file')
    parser.add_argument('--output', '-o', type=str, default = "./items.json",
                        help='Output path of the exported file')
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v, -vv for more detail)"
    )

    # Parse the arguments
    args = parser.parse_args(argv)
    anno2_path: str = args.anno2_path
    output_path: str = args.output
    logger = get_logger(args.verbose)

    logger.debug(f'Exporting anno2 at {anno2_path}, ')
    
    if not zipfile.is_zipfile(anno2_path):
        raise Exception(f"Anno2 @ {anno2_path} is not a zipfile, and could therefore not be a Anno2 file. Please check your input")
    anno2 = zipfile.ZipFile(anno2_path)
    decoder = Decoder(anno2, args.verbose)
    decoder.decode()

if __name__ == "__main__":
    main()
