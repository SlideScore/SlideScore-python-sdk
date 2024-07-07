import math
import array
from io import BufferedWriter, BytesIO
from typing import List

from .AnnoClasses import Polygons
from .omega_encoder import OmegaEncoder

"""
For storing a polygon (or multiple) 
for polygon with coords (on a slide <512x512 ie 2x2 tiles):
1,1
1,2
300,300
3,3

to store: (everything in omega)
tile lengths: 3, 0, 0, 1 // tile_lengths -- NOT USED
#points in tile 2, 1, 1 // num_points_in_tile, sums up to num points in polygon
xjump to next tile: 0,1,-1 
yjump to next tile  0,1,-1

+1byte remainders by div 256: 1, 1, 1, 2, 44, 44, 3, 3
"""

Polygon = List[int] # List/array like [x1, y1, x2, y2], above polygon would be [1, 1, 1, 2, 300, 300, 3, 3]

def encode_polygon(polygon: Polygon, tile_size: int):
    """Encode a list of flat vertices (polygon) into a tile-based format. Since the remainders of the vertices
    are stored in a raw byte array. The tile_size should not exceed 256."""
    last_tile_x = 0
    last_tile_y = 0

    num_points_in_tile = []
    x_jumps, y_jumps = [], []
    remainders = array.array('B') # Init byte array for remainders, so tile_size <= 256!

    for i in range(0, len(polygon), 2):
        point_x, point_y = polygon[i], polygon[i + 1]
        tile_x = math.floor(point_x / tile_size)
        tile_y = math.floor(point_y / tile_size)

        # Update jump tables

        # Check if tile changes in x or y direction, or first point
        if tile_x != last_tile_x or tile_y != last_tile_y or len(num_points_in_tile) == 0: 
            x_jump = tile_x - last_tile_x 
            y_jump = tile_y - last_tile_y
            x_jumps.append(x_jump)
            y_jumps.append(y_jump)

            last_tile_x = tile_x
            last_tile_y = tile_y
            
            num_points_in_tile.append(1) # At least this point is in the tile
        else: # Tile remains the same
            # So add one to the last num points in tile
            num_points_in_tile[-1] += 1

        # Add remainders
        remainders.extend((point_x % tile_size, point_y % tile_size))

    return x_jumps, y_jumps, num_points_in_tile, remainders

def calc_num_tile_rows_cols(polygon: Polygon, tile_size: int):
    """Finds the biggest point in a polygon and divides it by the tile size to find the number of needed
    tiles to encode all points."""
    max_x, max_y = 0, 0

    for i in range(0, len(polygon), 2):
        point_x, point_y = polygon[i], polygon[i + 1]
        max_x = max(point_x, max_x)
        max_y = max(point_y, max_y)
    
    num_rows = math.ceil(max_y / tile_size)
    num_cols = math.ceil(max_x / tile_size)

    return num_rows, num_cols

def concat_polygons(polygons: Polygons):
    """Concatenates/converts a "Polygons" object together into a single big polygon and a list of polygon sizes"""
    polygon_lengths = array.array('I')
    combined_polygon = polygons.polygons.valuesArray

    offsets = polygons.polygons.offsetArray
    for i in range(1, len(offsets)):
        last_offset = offsets[i - 1]
        offset = offsets[i]
        length = offset - last_offset
        polygon_lengths.append(length)
    
    return polygon_lengths, combined_polygon

def dump_array_enc(fh: BufferedWriter, nums: List[int], encoding_type: str):
    """Encodes an array of integers using Elias Omega encoding and writes the length of the resulting
    buffer and that buffer to a writer
    """
    encoder = OmegaEncoder()
    encoded_arr = encoder.encode(nums, encoding_type)
    fh.write(encoded_arr.nbytes.to_bytes(4, 'little'))
    fh.write(encoded_arr.tobytes())


def dump_2_disk(fh: BufferedWriter, polygon_lengths: array.ArrayType, tile_size: int, num_rows: int, num_cols: int, encoded_polygon):
    """Dumps an encoded polygon to disk/writer, writes lengths as unsigned 32 bit ints, including the polygon lengths array. 
    Omega encodes the other information, including tile_lengths, jumps etc."""
    # Write some metadata to the start of the file
    # TODO: Add a magic number to check if the file is valid?
    fh.write(tile_size.to_bytes(4, 'little')) # Should be 256
    fh.write(num_rows.to_bytes(4, 'little'))
    fh.write(num_cols.to_bytes(4, 'little'))
    
    # Write the length of each of the invidual polygons are uint_32t's to disk
    polygon_lengths_bytes = polygon_lengths.tobytes()
    fh.write(len(polygon_lengths_bytes).to_bytes(4, 'little'))
    fh.write(polygon_lengths_bytes)

    # Extract the encoded combined polygon into it's parts
    x_jumps, y_jumps, num_points_in_tile, remainders = encoded_polygon

    # Dump the data omega encoded to disk
    dump_array_enc(fh, x_jumps, 'integers') # Can contain negative ints
    dump_array_enc(fh, y_jumps, 'integers') # ^^
    dump_array_enc(fh, num_points_in_tile, 'naturalOnly') # Always >= 1

    # Finally write the coordinate % 256 remainders to disk
    fh.write(len(remainders).to_bytes(4, 'little'))
    fh.write(remainders.tobytes())

def polygons_2_bytes(polygons: Polygons, tile_size = 256):
    """Encodes polygons and dumps them to a byte array"""

    polygon_lengths, combined_polygon = concat_polygons(polygons)

    num_rows, num_cols = calc_num_tile_rows_cols(combined_polygon, tile_size)

    encoded_polygon = encode_polygon(combined_polygon, tile_size)

    write_handle = BytesIO()
    dump_2_disk(write_handle, polygon_lengths, tile_size, num_rows, num_cols, encoded_polygon)
    return write_handle.getbuffer()

if __name__ == "__main__":
    print("Generating simple encoded polygons!")
    polygons = Polygons()
    polygon = [1, 1, 1, 2, 300, 300, 3, 3]
    polygons.addPolygon(polygon)
    polygon = [400, 400, 200, 200, 5, 5, 3, 3, 0, 0]
    polygons.addPolygon(polygon)
    
    buf = polygons_2_bytes(polygons)
    with open('encoded_polygons.bin', 'wb') as fh:
        fh.write(buf)
    print("Wrote encoded_polygons.bin")
