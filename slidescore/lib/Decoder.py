import math
import array
import json
import os
import tarfile
import typing
import zipfile
import io
from typing import List, Dict, Any
import zipfile
from datetime import datetime, timezone
from packaging import version

from bitarray import bitarray
import msgpack
import brotli
import png

from slidescore.lib.omega_encoder import OmegaEncoder

from .utils import get_logger
from .AnnoClasses import Polygons, Points, Items

logger = get_logger(0)

class Decoder():
    """Encompassing class to decode Anno2"""

    supported_types = ['polygons', 'points', 'mask'] # mask is processed the same way as points
    anno2_version = version.parse("0.0.0")
    anno2_type = ''
    system_metadata = None
    items: Items = None

    def __init__(self, anno2: zipfile.ZipFile, verbosity = 0) -> None:
        """Initialize decoder with a Anno2 zipfile, checking if we can convert it
        """
        global logger
        logger = get_logger(verbosity)

        self.anno2 = anno2
        self._check_if_compatible_anno2()
        logger.notice(f"Loaded Anno2 (v{self.anno2_version}, type={self.system_metadata['type']}, numItems={self.system_metadata['numItems']})")

    def _check_if_compatible_anno2(self):
        """Also loads anno2_version & system_metadata"""
        try:
            with self.anno2.open('system_metadata.json') as f:
                logger.debug(f"Detected system_metadata.json exists in the zip!")
                self.system_metadata = json.load(f)

                # Check version
                ver_str = self.system_metadata['version']
                v = version.parse(ver_str)
                self.anno2_version = v
                # Check if version is below 1.0.0
                if v >= version.parse("1.0.0"):
                    raise ValueError(f"Anno2 version {ver_str} must be below 1.0.0")

                # Log a warning if version is not exactly 0.2.0
                if v != version.parse("0.2.0"):
                    logger.warning(f"Version {ver_str} is not exactly 0.2.0, contineuing on the presumption the anno2 file is compatible")

                # Check the type
                self.anno2_type = self.system_metadata['type']
                if self.anno2_type not in self.supported_types:
                    raise ValueError(f"Anno2 type '{self.anno2_type}' not in supported types {self.supported_types}")
        except KeyError:
            raise Exception('Could not detect system_metadata.json in the Anno2 Zipfile, please check your input')

    def decode(self):
        logger.debug("Starting decode")
        if self.anno2_type == 'polygons':
            self.items = self._decode_polygons()
        elif self.anno2_type == 'points' or self.anno2_type == 'mask':
            self.items = self._decode_points()
        else:
            raise TypeError(f'Anno2 type {self.anno2_type=} not recognized')
        
        assert self.items is not None
        
        if type(self.system_metadata['numItems']) is int:
            if self.system_metadata['numItems'] == len(self.items):
                logger.notice(f'Decoded anno2 had the correct number of items ({len(self.items)})')
            else:
                logger.warning(f"Decoded anno2 had an unexpected number of items ({self.system_metadata['numItems']} != {len(self.items)})")
        else:
            logger.debug('Could not verify the numItems because it is not an int')

        logger.debug("End decode")

    def dump_to_file(self, path: str):
        """Dumps the decoded items to a file on disk, file format depends on the . Also encodes the polygon if needed."""
        logger.debug("Encoding and dumping to file")

        # Find the used output type
        preffered_output_type = self._infer_output_type(path)
        all_supported_output_types = {
            Polygons: ['json', 'tsv', 'geojson'],
            Points: ['json']
        }

        if not type(self.items) in all_supported_output_types:
            raise ValueError(f'Failed to find {type(self.items)=} among supported output types')

        cur_supported_output_types = all_supported_output_types[type(self.items)]
        if preffered_output_type in cur_supported_output_types:
            output_type = preffered_output_type
            logger.info(f'Was able to select {preffered_output_type} as output type')
        else:
            output_type = cur_supported_output_types[0]
            logger.warning(f'Was not able to select the detected preffered output type ({preffered_output_type}) as output type, using {output_type}')

        # Dump the items data to the detected available output type
        if output_type == 'json':
            anno1_obj = self._items_to_anno1_obj()
            with open(path, 'w') as output_fh:
                json.dump(anno1_obj, output_fh)
        elif output_type == 'tsv':
            with open(path, 'w') as output_fh:
                self._write_items_to_tsv(output_fh)
        elif output_type == 'geojson':
            anno1_obj = self._items_to_geojson_obj()
            with open(path, 'w') as output_fh:
                json.dump(anno1_obj, output_fh)

        logger.info(f'Dumped {type(self.items)=} to {path=}')

    def _items_to_anno1_obj(self):
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        anno1_object = []

        if type(self.items) is Polygons:
            for i in range(len(self.items)):
                anno1_polygon = {
                    "type": 'polygon',
                    "modifiedOn": timestamp,
                    "points": []
                }
                polygon = self.items[i]
                for x, y in polygon['positiveVertices']:
                    anno1_polygon['points'].append({ "x": x, "y": y })
                anno1_object.append(anno1_polygon)
        elif type(self.items) is Points:
            anno1_points = [self.items[i] for i in range(len(self.items))]
            return anno1_points
        else:
            raise TypeError(f'Type {type(self.items)} not yet implemented for conversion to Anno1')
        return anno1_object

    def _write_items_to_tsv(self, fh: typing.TextIO):
        if type(self.items) is Polygons:
            for i in range(len(self.items)):
                polygon = self.items[i]
                for j in range(len(polygon['positiveVertices'])):
                    x, y = polygon['positiveVertices'][j]
                    is_last = j == len(polygon['positiveVertices']) - 1
                    logger.debug(f'{x=} {y=} {is_last=}')
                    fh.write(f'{x}\t{y}' + ('\t' if not is_last else ''))
                fh.write('\n')
        else:
            raise TypeError(f'Type {type(self.items)} not yet implemented for conversion to TSV')
        return None

    def _items_to_geojson_obj(self):
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        geojson_features = []
        geojson_object = {
            "type": "FeatureCollection",
            "features": geojson_features
        }

        if type(self.items) is Polygons:
            for i in range(len(self.items)):
                points = [] # [[x1, y1], [x2, y2], etc...]
                geojson_polygon = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ points ]
                    },
                    "properties": {
                        "exported_at": timestamp,
                        "exported_from": "slidescore-anno2"
                    }
                }
                polygon = self.items[i]

                # We do not re-order the points to follow the right hand rule,
                # because that order would not have been present in the original dataset
                for x, y in polygon['positiveVertices']:
                    points.append([x, y])
                geojson_features.append(geojson_polygon)
        else:
            raise TypeError(f'Type {type(self.items)} not yet implemented for conversion to GeoJSON')
        return geojson_object

    def _infer_output_type(self, path: str):
        """
        Parse a path and return a normalized preferred output type.

        Supported:
            - .tsv
            - .json
            - .png
            - .geo.json / .geojson

        Returns:
            A string representing the normalized output type, e.g.:
                "tsv", "json", "png", "geojson", "unknown"
        """
        filename = os.path.basename(path).lower()

        # Handle .geo.json (two-suffix case)
        if filename.endswith(".geo.json"):
            return "geojson"

        # Get last extension
        ext = os.path.splitext(filename)[1]

        # Normalize extension (strip leading dot)
        ext = ext.lstrip(".")

        # Map simple extensions
        if ext in {"tsv", "json", "png", "geojson"}:
            return ext

        return "unknown"

    def _decode_points(self):
        logger.debug("Starting decode of points (or mask)")
        """
        Points have 2 encoding options
            if there are few points or the density is low,
                a brotli encoded Anno1 JSON is simply stored
            If there are many points a masks.tar.gz is created with tile_x{tile_x}_y{tile_y}.png files
                this contains a black and white png with the points (aka a mask)
        """
        points = Points()
        try:
            with self.anno2.open('anno1_points.json.br') as f:
                anno1_points_bytes: bytes = brotli.decompress(f.read())
                anno1_points = json.loads(anno1_points_bytes)
                for point in anno1_points:
                    x, y = point['x'], point['y']
                    points.addPoint(x, y)
            return points
        except KeyError:
            # if the points are stored as a mask
            with self.anno2.open('masks.tar.gz') as f:
                with tarfile.open(fileobj=f, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        logger.debug(f"Name: {member.name}, Size: {member.size} bytes")
                        
                        # If you want to read the content of a file
                        if member.isfile():
                            file_content = tar.extractfile(member).read()
                            if file_content[:4] == b'\x89PNG':
                                # It's a PNG
                                assert member.name.startswith('tile_')
                                tile_name_parts = member.name.split('_')
                                logger.debug(f'{tile_name_parts=}')
                                tile_x = int(tile_name_parts[1][1:])
                                tile_y = int(tile_name_parts[2][1:].removesuffix('.png'))
                                
                                # tile_size should always be 256 for now, perhaps different in newer anno2 versions
                                tile_size, png_points = self._decode_png(file_content)
                                for x, y in png_points:
                                    img_x = tile_x * tile_size + x
                                    img_y = tile_y * tile_size + y
                                    logger.debug(f'PNG point decoded: {img_x}, {img_y}')
                                    points.addPoint(img_x, img_y)
            return points


    def _decode_polygons(self):
        logger.debug("Starting decode of polygons")
        """
        Zipfile contains
        polygon_container/
            big_tile_polygons_i.msgpack.br - Not needed
            encoded_polygons.bin.br        - Contains the raw polygon coords remains
            negative_polygons.json         - Always empty json for now
            simpl_encoded_polygons.bin.br  - Contains presimplified, not needed
            tile_polygons_i.msgpack.br     - Contains the indices of polygons in the tile (256x256 px)
        """
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
        with self.anno2.open('polygon_container/encoded_polygons.bin.br') as f:
            logger.debug(f"Starting decode of encoded_polygons.bin.br")
            encoded_polygons_bytes: bytes = brotli.decompress(f.read())
            buf = memoryview(encoded_polygons_bytes)
        logger.debug("Read encoded polygons from Anno2")

        tile_size = int.from_bytes(buf[:4], 'little')
        if tile_size != 256:
            raise ValueError(f'Did not read expected tile size from polygons binary format {tile_size=} != 256, is your file corrupted?')
        num_rows = int.from_bytes(buf[4:8], 'little')
        num_cols = int.from_bytes(buf[8:12], 'little')
        polygon_lengths_byte_len = int.from_bytes(buf[12:16], 'little')

        polygon_lengths_bytes = buf[16:16 + polygon_lengths_byte_len]
        pos = 16 + polygon_lengths_byte_len

        polygon_lengths = array.array('I')
        polygon_lengths.frombytes(polygon_lengths_bytes)
        logger.info(f"{tile_size=} {num_rows} {num_cols=} {polygon_lengths_byte_len=} {polygon_lengths=}")
        
        if len(polygon_lengths) != self.system_metadata['numItems']:
            logger.warning(f"Did not decode the expected number ({self.system_metadata['numItems']}) of polgons, got {len(polygon_lengths)}")
        else:
            logger.debug(f"Got the expected number of polygons {len(polygon_lengths)=}")
        
        # Now we need to decode the omega encoded x_jumps, y_jumps, num_points_in_tile
        x_jumps,            pos = self._decode_omega_encoded_array(buf, pos, 'integers')
        y_jumps,            pos = self._decode_omega_encoded_array(buf, pos, 'integers')
        num_points_in_tile, pos = self._decode_omega_encoded_array(buf, pos, 'naturalOnly')
        logger.debug(f"{x_jumps=} {y_jumps=} {num_points_in_tile=}")
        remainders_len = int.from_bytes(buf[pos:pos + 4], 'little')
        pos += 4
        remainders = buf[pos:pos + remainders_len]
        logger.debug(f'{remainders_len=} {list(remainders.cast("B"))=}')

        # Perform final decode step
        raw_polygons = self._polygons_recombine_into_raw(x_jumps, y_jumps, num_points_in_tile, tile_size, remainders, polygon_lengths)
        logger.info(f'{raw_polygons[:5]=}')
        return raw_polygons

    def _decode_omega_encoded_array(self, buf: memoryview, pos: int, type: str):
        """Decode omega encoded array, can contain suffixing zeros/ones due to byte padding"""
        omega_decoder = OmegaEncoder()
        array_byte_count = int.from_bytes(buf[pos:pos + 4], 'little')
        pos += 4
        array_bytes = buf[pos:pos + array_byte_count]
        pos += array_byte_count
        array_bitarray = bitarray()
        array_bitarray.frombytes(array_bytes)
        array = omega_decoder.decode(array_bitarray, type)
        return array, pos

    def _polygons_recombine_into_raw(
            self,
            x_jumps: List[int],
            y_jumps: List[int],
            num_points_in_tile: List[int],
            tile_size: int,
            remainders: memoryview,  # uint8
            polygon_lengths: array.array  # array of type 'I' (uint32)
    ):
        tile_x = 0
        tile_y = 0

        num_points = len(remainders) // 2
        
        polygon_lengths = polygon_lengths[:] # make a copy because we are going to modify it
        polygons = Polygons()
        cur_polygon = []

        remainder_i = 0
        num_jumps = min(len(num_points_in_tile), len(x_jumps), len(y_jumps))
        for i in range(num_jumps):
            x_jump = x_jumps[i]
            y_jump = y_jumps[i]

            tile_x += x_jump
            tile_y += y_jump

            logger.debug(f'{len(num_points_in_tile)=}, {len(x_jumps)=} {i=}')
            num_points_in_this_tile = num_points_in_tile[i]
            for _ in range(num_points_in_this_tile):
                if remainder_i // 2 >= num_points:
                    # Because the jump tables are stored as a bit array with suffixing zeros,
                    # these are mistaken for encoded 1's, so we stop if we run out of points
                    break

                x_in_tile = remainders[remainder_i]
                y_in_tile = remainders[remainder_i + 1]

                raw_x = tile_x * tile_size + x_in_tile
                raw_y = tile_y * tile_size + y_in_tile

                cur_polygon.extend([raw_x, raw_y])
                if len(cur_polygon) == polygon_lengths[0]:
                    polygons.addPolygon(cur_polygon)
                    polygon_lengths = polygon_lengths[1:]
                    cur_polygon = []

                remainder_i += 2

        return polygons

    def _decode_png(self, png_buf: bytes):
        """Decodes a PNG into a tile_size and points list"""
        reader = png.Reader(bytes=png_buf)
        width, height, rows, info = reader.read()
        assert width == height
        points = []
        if info['bitdepth'] == 1:
            # Extract the points from the rows
            row_i = 0
            for row in rows:
                x_vals = [i for i, b in enumerate(row) if b != 0]
                for x in x_vals:
                    points.append((x, row_i))
                row_i += 1
        logger.debug(f'PNG {width=} {height=} {points=} {info=}')
        return width, points