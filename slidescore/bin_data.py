DESC = """
This program converts a items TSV file (or slidescore_anno1.json) of either points in a mask, polygons or a heatmap, into a binned format for fast lookup. 
Author: Bart.
"""

import sys
import argparse
import json
import gzip

from .lib.utils import log, read_geo_json, read_tsv, read_slidescore_json
from .lib.Encoder import Encoder

def main(argv=None):
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('--items-path', '-i', type=str, required=True,
                        help='Input file path, should be a TSV / GeoJSON / SlideScore JSON file')
    parser.add_argument('--output', '-o', type=str, default = "./items_binned.zip",
                        help='Output path of the binned items file')
    parser.add_argument('--metadata', '-m', type=str,
                        help='Path of JSON file containing user-specified metadata about the input file, will be encoded in the output file')
    parser.add_argument('--points-type', '-pt', choices=['mask', 'circles'], default="circles",
                        help='Type of points that are provided in the TSV, either single pixels (mask), or center points of circles (default)')

    # Parse the arguments
    args = parser.parse_args(argv)
    raw_items_path = args.items_path
    binned_items_path: str = args.output

    log('Reading data into memory')
    
    # Parse the input items
    if raw_items_path.endswith('.tsv'):
        items = read_tsv(raw_items_path, args.points_type)
    elif raw_items_path.endswith('.geojson'):
        items = read_geo_json(raw_items_path)
    elif raw_items_path.endswith('.json'):
        with open(raw_items_path, 'r') as fh:
            data = json.load(fh)
            items = read_slidescore_json(data)
    elif raw_items_path.endswith('.json.gz'):
        with gzip.open(raw_items_path, 'r') as fh:
            data = json.load(fh)
            items = read_slidescore_json(data)
    else:
        sys.exit("Please provide a .tsv/.geojson file")
    
    log('Loaded data into memory')

    encoder = Encoder(items, big_polygon_size_cutoff=100 * 100)
    encoder.generate_tile_data(256)
    log('Binned items into 256x256 tiles')
    
    encoder.populate_lookup_tables()
    log('Generated lookup tables')
    if args.metadata:
        with open(args.metadata, 'r') as fh:
            metadata = json.load(fh)
            encoder.add_metadata(metadata)

    encoder.dump_to_file(binned_items_path)
    log('Dumped to file')

if __name__ == "__main__":
    main()
