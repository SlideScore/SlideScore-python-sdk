import math
import array
import json
import tarfile
import zipfile
import io
from typing import List, Dict, Any

import msgpack
import brotli

from .image_utils import get_png_bytes, lookup_table_2_png, encode_png
from .AnnoClasses import Points, Polygons, Heatmap, Items, Item
from .PolygonContainer import PolygonContainer
from .utils import log, msgpack_encoder

class Encoder():
    """Encompassing class to encode AnnoClasses"""
    items: Items = None

    dataItems: Dict[str, Any] = None # Data regarding the raw mask / polygon / heatmap containers
    dataLookup: List[Dict] = None # Data regarding lookup tables for mask and polygons
    system_metadata = {} # Metadata that is controlled by this encoder, used in decoding
    user_metadata = {} # Metadata that the user passes regarding this annotation, saved in the output zip
    big_polygon_size_cutoff = 100 * 100 # Size that is considered a "big" polygon, saved seperatly
    few_points_cutoff = 500 * 1000 # Determined using some tests
    low_density_cutoff = 30 # If there are fewer than 30 points per 256x256 PNG tile, save points as json

    def __init__(self, items: Items, big_polygon_size_cutoff=100 * 100) -> None:
        """Initialize encoder with a list of either Points or Polygons or Heatmap
        
        A point should be a tuple of (image_x, image_y).
        A polygon should be a dictionary with "positiveVertices" containing a list of points.
            And possible a key "negativeVertices" contain a seperate polygon"""
        self.items = items
        self.big_polygon_size_cutoff = big_polygon_size_cutoff
        self.dataItems = {
            "numItems": len(items)
        }
        self.dataLookup = []

        type_string = items.name.lower() # can be points/mask/polygons/heatmap
        self.system_metadata = {
            "version": "0.0.3",
            "type": type_string,
            "numItems": len(items)
        }
        self.user_metadata = {}

        if isinstance(items, Points):
            log("Loaded", self.dataItems["numItems"], "points in encoder, type:", type_string)
        elif isinstance(items, Polygons):
            num_points = int(len(items.polygons.valuesArray) / 2)
            log("Loaded", self.dataItems["numItems"], "polygons in encoder, with num points", num_points)
        elif isinstance(items, Heatmap):
            log("Loaded", self.dataItems["numItems"], "byte heatmap in encoder, with shape", items.matrix.shape)



    def calc_rect_around_item(self, item: Item):
        """Calculates a bounding box around a point or polygon, if it is a point the rectangle is the point"""
        if isinstance(self.items, Points):
            item = [item] # Pack it like a polygon with 1 point
        else:
            item = item["positiveVertices"] # Extract the positive vertices
        # Save the most extreme x and y values
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        for point in item:
            x, y = point
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)
        return min_x, min_y, max_x, max_y

    def calc_tile_range(self, min_x: float, min_y: float, max_x: float, max_y: float, tile_size: int):
        """Calculates the tiles that intersect with a bounding box"""
        return {
            "x": {
                "start": math.floor(min_x / tile_size),
                "end": math.floor(max_x / tile_size),
            },
            "y": {
                "start": math.floor(min_y / tile_size),
                "end": math.floor(max_y / tile_size)
            }
        }
    
    def get_polygon_size(self, item: Item):
        """Calculates the area the bounding box around a polygon covers"""
        polygon_rect = self.calc_rect_around_item(item)
        min_x, min_y, max_x, max_y = polygon_rect
        width = max_x - min_x
        height = max_y - min_y
        return width * height

    def get_tiles_containing_item(self, item: Item, tile_size: int):
        """Calculates which tile indices contain a point or polygon"""
        polygon_rect = self.calc_rect_around_item(item)
        min_x, min_y, max_x, max_y = polygon_rect
        # Find the tiles this rect is in
        tile_range = self.calc_tile_range(min_x, min_y, max_x, max_y, tile_size)
        return tile_range

    def generate_tile_data(self, tile_size = 256):
        """Bin the loaded items (points/polygons) into seperate tiles. Optionally encodes these
        tiles into PNG's for points, and keeps track of big polygons in the tiles."""

        # Save the tile_bins in the appropriate location
        if isinstance(self.items, Points):
            # Convert the points into masks
            self.dataItems["masks"] = self.bin_points_into_tiles(tile_size)
        elif isinstance(self.items, Polygons):
            self.dataItems["polygonContainer"] = self.bin_polygons_into_tiles(tile_size)
        elif isinstance(self.items, Heatmap):
            shape = self.items.matrix.shape
            self.dataItems["heatmapPng"] = encode_png(self.items.matrix, shape[1], shape[0], bitdepth=8)


    def bin_points_into_tiles(self, tile_size):
        """Bins the loaded items into a tile-based format. These are basically masks.
        So this format is generally referred to as masks.
        
        Uses a dictionary of tile indices data[tile_y][tile_x] = list of points in tile.
        Encodes the tiles as PNG's for more effecient storage.
        """
        items = self.items
        
        # Determine if we want to store the points as a compressed JSON, or as PNG tiles
        are_few_points = len(items) < self.few_points_cutoff
        is_points = self.items.name == 'points' # Could also be mask
        has_metadata = 'metadata' in self.items

        if (are_few_points and is_points) or (has_metadata and is_points):
            log(f"Detected few points ({len(items)}), saving anno1 JSON")
            # Instead encode as a Anno1 JSON, that will get compressed when dumping to a file
            anno1_points = [{"x": img_x, "y": img_y} for img_x, img_y in items]
            return json.dumps(anno1_points)

        tile_bins = {}
        num_tiles = 0
        for point in items:
            # Calculate the tile this point is in
            img_x, img_y = point
            tile_x = math.floor(img_x / tile_size)
            tile_y = math.floor(img_y / tile_size)
            
            # Create tile lookup if not present yet
            if tile_y not in tile_bins:
                tile_bins[tile_y] = {}
            if tile_x not in tile_bins[tile_y]:
                # Make array to hold the points
                tile_bins[tile_y][tile_x] = []
                num_tiles += 1
            
            # Add point to tile
            new_point = (img_x % tile_size, img_y % tile_size)
            tile_bins[tile_y][tile_x].extend(new_point)

        # If we detect that the density of points is low, e.g. < 30 points per 256x256 tile,
        # encode as JSON anyway
        num_points_per_tile = len(items) / num_tiles
        if num_points_per_tile < self.low_density_cutoff and is_points:
            log(f"Detected low density of points ({round(num_points_per_tile)} / tile), saving anno1 JSON")
            anno1_points = [{"x": img_x, "y": img_y} for img_x, img_y in items]
            return json.dumps(anno1_points)


        # When we are done binning all points into the tiles, compress the tiles
        # into PNGs. A single PNG is a mask.
        log("Compressing tiles as png's", num_tiles, len(items), len(items) / num_tiles)
        for tile_y in tile_bins:
            for tile_x in tile_bins[tile_y]:
                tile = tile_bins[tile_y][tile_x]
                img_bytes = get_png_bytes(tile, tile_size)
                tile_bins[tile_y][tile_x] = img_bytes
        log("Done compressing tiles")
        return tile_bins

    def bin_polygons_into_tiles(self, tile_size: int):
        """Similar to bin_points_into_tiles, this bins the polygons into tiles. It uses the bounding box
        around a polygon to estimate where to bin them in. Also stores the big polygons in a seperate list."""
        items = self.items
        tile_bins = PolygonContainer(tile_size, items)

        for i in range(len(items)):
            polygon = items[i]
            polygon_size = self.get_polygon_size(polygon)
            is_big_polygon = polygon_size > self.big_polygon_size_cutoff
            tile_range = self.get_tiles_containing_item(polygon, tile_size)

            # Add the polygon to the container
            tile_bins.store_polygon_i(i, tile_range, is_big_polygon)
            del polygon
        
        return tile_bins

    # Lookup table generation
    def populate_lookup_tables(self):
        """Bins items into seperate "tiles", which are then used to create a density map PNG to give a simplified representation
        of the items that can easily be process/drawn. """
        if isinstance(self.items, Heatmap):
            log("Skipping lookup table generation for heatmap")
            return
        
        if len(self.items) < self.few_points_cutoff and self.items.name == 'points':
            log("Skipping lookup table generation for few points")
            return

        for tile_size in [32, 256]:
            # Check if we can use the fast path using an old lookup table
            fast_path_option = next(filter(lambda lookupData: tile_size % lookupData["tile_size"] == 0, self.dataLookup), None)

            if fast_path_option:
                lookup_table = self.bin_items_into_lookup_table_fast(tile_size, fast_path_option["tile_size"], fast_path_option["lookup"]) 
            else:
                lookup_table = self.bin_items_into_lookup_table(tile_size) 
            
            lookup_table['png'] = lookup_table_2_png(lookup_table)
            self.dataLookup.append(lookup_table)
            log("Done with lookup table of size", tile_size)

    def bin_items_into_lookup_table(self, tile_size: int):
        """Method to bin items into "tiles" of a certain size and keep track on how many items fall inside such a tile.
        Used to create density maps."""
        items = self.items
        # Need to add a factor for polygons because 1 polygon paints many pixels
        num_points_to_add = 1 if isinstance(self.items, Points) else 15

        tile_bins = {}
        data = {
            "tile_size": tile_size,
            "lookup": tile_bins,
            "maxValue": 0,
        }

        for i in range(len(items)):
            item = items[i]

            # Ignore big polygons for lookup generation
            if isinstance(items, Polygons):
                polygon_size = self.get_polygon_size(item)
                is_big_polygon = polygon_size > self.big_polygon_size_cutoff
                if is_big_polygon:
                    continue

            tile_range = self.get_tiles_containing_item(item, tile_size)
            
            # Add polygon to every tile present in this tile_range (add one since range is inclusive)
            for yI in range(tile_range["y"]["start"], tile_range["y"]["end"] + 1):
                for xI in range(tile_range["x"]["start"], tile_range["x"]["end"] + 1):
                    # Create tile lookup if not present yet
                    if yI not in tile_bins:
                        tile_bins[yI] = {}
                    if xI not in tile_bins[yI]:
                        tile_bins[yI][xI] = 0
                    
                    # Add one to it
                    
                    tile_bins[yI][xI] += num_points_to_add
                    data["maxValue"] = max(data["maxValue"], tile_bins[yI][xI])

        return data

    def bin_items_into_lookup_table_fast(self, new_tile_size: int, old_tile_size: int, old_lookup_table):
        """Faster method to create lookup tables for density maps. By using a previously determined smaller tile
        lookup entry, you can easily sum the small tiles that fall inside the big tile."""
        if new_tile_size % old_tile_size != 0:
            raise Exception('Cannot use fast method')
    
        tile_index_ratio = new_tile_size / old_tile_size # Always an int as checked above
        
        tile_bins = {}
        data = {
            "tile_size": new_tile_size,
            "lookup": tile_bins,
            "maxValue": 0,
        }

        for yI in old_lookup_table:
            row = old_lookup_table[yI]
            for xI in row:
                num_points = row[xI]
                # Calculate the indices in the new lookup table
                new_yI = math.floor(yI / tile_index_ratio)
                new_xI = math.floor(xI / tile_index_ratio)

                # Create tile lookup if not present yet
                if new_yI not in tile_bins:
                    tile_bins[new_yI] = {}
                if new_xI not in tile_bins[new_yI]:
                    tile_bins[new_yI][new_xI] = 0

                tile_bins[new_yI][new_xI] += num_points
                data["maxValue"] = max(tile_bins[new_yI][new_xI], data["maxValue"])
        return data


    def dump_to_file(self, path: str):
        """Dumps the encoded items to a ZIP file on disk. Also encodes the polygon if needed."""
        log("Encoding and dumping to zipfile")

        if not path.endswith('.zip'):
            path += '.zip'

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zip:
            # Add metadata files
            system_metadata_bytes = str.encode(json.dumps(self.system_metadata, indent=2))
            zip.writestr('system_metadata.json', system_metadata_bytes)

            user_metadata_bytes = str.encode(json.dumps(self.user_metadata, indent=2))
            zip.writestr('user_metadata.json', user_metadata_bytes)

            # Add lookup tables
            for lookup_data in self.dataLookup:
                tile_size = lookup_data["tile_size"]
                png_bytes = lookup_data["png"]

                # Add the png
                zip.writestr(f'lookup-tables/density_{tile_size}px.png', png_bytes)

            
            # Add mask data
            if 'masks' in self.dataItems:
                if isinstance(self.dataItems["masks"], dict):
                    tar_gz_fh = io.BytesIO()
                    with tarfile.open(fileobj=tar_gz_fh, mode="w:gz") as tar:
                        tile_bins = self.dataItems["masks"]
                        for tile_y in tile_bins:
                            for tile_x in tile_bins[tile_y]:
                                tile_png_bytes = tile_bins[tile_y][tile_x]
                                fn = f'tile_x{tile_x}_y{tile_y}.png'
                                add_buffer_2_tar(tar, tile_png_bytes, fn)
                    zip.writestr(f'masks.tar.gz', tar_gz_fh.getbuffer())
                else:
                    zip.writestr(f'anno1_points.json.br', brotli.compress(self.dataItems["masks"].encode(), quality=8))
            
            # Add polygons if possible
            if 'polygonContainer' in self.dataItems:
                add_polygon_container_2_zip(zip, self.dataItems["polygonContainer"], 'polygon_container')
            # Add any labels if present
            if isinstance(self.items, Polygons) and len(self.items.labels) > 0:
                labels_bytes = str.encode(json.dumps(self.items.labels, indent=2))
                zip.writestr('labels.json', labels_bytes)

            # Add heatmap if available
            if 'heatmapPng' in self.dataItems:
                zip.writestr(f'heatmap.png', self.dataItems['heatmapPng'])
                # Add heatmap metadata
                heatmap_metadata_bytes = str.encode(json.dumps(self.items.get_metadata(), indent=2))
                zip.writestr('heatmap_metadata.json', heatmap_metadata_bytes)

    def add_metadata(self, metadata):
        """Stores a metadata object that gets copied into the output file"""
        self.user_metadata = metadata

def flat(items):
    """Flattens list from [[1, 2], [3, [4]]] -> [1, 2, 3, [4]]"""
    return [item for sublist in items for item in sublist]


def add_polygon_container_2_zip(zip, container, dir_name):
    """Utility to encode a polygon container class and add it to a ZIP file"""
    # Start with the polygon indices in each tile
    tile_polygons_i = container.allTiles
    tile_polygons_i_bytes = msgpack.dumps(tile_polygons_i, default=msgpack_encoder)
    tile_polygons_i_compressed_bytes = brotli.compress(tile_polygons_i_bytes, quality=8)

    zip.writestr(f'{dir_name}/tile_polygons_i.msgpack.br', tile_polygons_i_compressed_bytes)

    # Also store the big polygon indices in each tile
    big_tile_polygons_i = container.bigTiles
    big_tile_polygons_i_bytes = msgpack.dumps(big_tile_polygons_i, default=msgpack_encoder)
    big_tile_polygons_i_compressed_bytes = brotli.compress(big_tile_polygons_i_bytes, quality=8)

    zip.writestr(f'{dir_name}/big_tile_polygons_i.msgpack.br', big_tile_polygons_i_compressed_bytes)

    # Then add the (compressed) encoded polygon bytes
    polygon_bytes = container.encode_polygons()
    polygon_compressed_bytes = brotli.compress(polygon_bytes, quality=8)
    zip.writestr(f'{dir_name}/encoded_polygons.bin.br', polygon_compressed_bytes)

    # Finally add the negative polygons
    negative_polygons_bytes = str.encode(json.dumps(container.polygons.negative_polygons_i, indent=2))
    zip.writestr(f'{dir_name}/negative_polygons.json', negative_polygons_bytes)

    # Add metadata if available
    if len(container.polygons.metadata) > 0:
        add_item_metadata_2_zip(zip, len(container.polygons), container.polygons.metadata, f'{dir_name}/items_metadata.jsonl')

def add_buffer_2_tar(tar, buffer, name):
    """Utility to add a buffer to a TARball with a certain name. Used to encode the points tile PNG's"""
    tarInfo = tarfile.TarInfo(name=name)
    tarInfo.size = len(buffer)
    tar.addfile(
        tarInfo,
        io.BytesIO(buffer)
    )

def add_item_metadata_2_zip(zip, num_items, item_metadata, fn):
    """
    Utility to add item metadata to an anno2 zip, item_metadata should be in the form:
    {2: {"note": "Example"}, 4: "color": "#FF00EE"}}. Where the integers represent item indices.
    """
    # Create a jsonl and an index of it
    jsonl = '\n'
    
    index = b''

    # Loop over all items
    for i in range(num_items):
        metadata = item_metadata.get(i)
        if not metadata:
            index += (0).to_bytes(4, byteorder='little', signed=False)
        else:
            index += len(jsonl).to_bytes(4, byteorder='little', signed=False)
            jsonl += json.dumps(metadata) + '\n'
    index += len(jsonl).to_bytes(4, byteorder='little', signed=False)

    jsonl_compressed_bytes = brotli.compress(str.encode(jsonl), quality=8)
    zip.writestr(fn + '.br', jsonl_compressed_bytes)

    index_compressed_bytes = brotli.compress(index, quality=8)
    zip.writestr(fn + '.index.br', index_compressed_bytes)