DESC = """
This program converts the Anno2 binned file to a common data format, like TSV, GeoJSON, or PNG. 
A preffered output type can be given by passing a output path with an appropriate extension:
.json for Anno1 JSON,
.geo.json / .geojson for GeoJSON,
the others as expected.

Be warned that this is not guaranteed, and the output file format should be checked manually.
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
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path of the exported file')
    parser.add_argument('--metadata-output', '-m', type=str, required=False,
                        help='Output path of the user supplied metadata json (optional)')
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
    logger.notice('Decoded items succesfully, saving to disk')
    decoder.dump_to_file(output_path)
    if args.metadata_output:
        decoder.dump_user_metadata_to_file(args.metadata_output)
    logger.notice('Finished saving to disk')

if __name__ == "__main__":
    main()
