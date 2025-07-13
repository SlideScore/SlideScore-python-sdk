import array
from typing import Dict

from .polygon_encoder import polygons_2_bytes
from .AnnoClasses import Polygons


class PolygonContainer:
    """Container of polygons and the tiles that contain these respective polygons.
    
    Every polygon is stored in a single list, and their index is stored in all tiles that might contain them
    When saving to disk, use the encode_polygons method to compress the stored polygons into a space-effecient format"""

    allTiles = {}
    bigTiles = {}
    polygons: Polygons = None
    tile_size = 256
    
    def __init__(self, tile_size: int, polygons: Polygons):
        self.allTiles = {}
        self.bigTiles = {}
        self.polygons = polygons
        self.tile_size = tile_size

    def store_polygon_i(self, polygon_i, tile_range: Dict[str, Dict[str, int]], storeInBig: bool):
        """Stores a polygon to a tile in a tile object
        
        polygon_i: index of self.polygons[i]
        tile_range: Dict with the start and end indices of the tiles containing the polygon
        storeInBig: Boolean to indicate whether this is a "big" polygon and index should be stored in that map
        """
        # Add the index of the polygon to all tiles that are in it's bounding box
        for tile_y in range(tile_range["y"]["start"], tile_range["y"]["end"] + 1):
            for tile_x in range(tile_range["x"]["start"], tile_range["x"]["end"] + 1):
                
                # Create the tiles if they do not exist yet
                if tile_y not in self.allTiles:
                    self.allTiles[tile_y] = {}
                if tile_x not in self.allTiles[tile_y]:
                    self.allTiles[tile_y][tile_x] = array.array('I')

                if storeInBig and tile_y not in self.bigTiles:
                    self.bigTiles[tile_y] = {}
                if storeInBig and tile_x not in self.bigTiles[tile_y]:
                    self.bigTiles[tile_y][tile_x] = array.array('I')
    
                # Add the polygon index to the tiles
                self.allTiles[tile_y][tile_x].append(polygon_i)
                if storeInBig:
                       self.bigTiles[tile_y][tile_x].append(polygon_i)

    def encode_polygons(self):
        """Encode the stored polygons in a very space effecient manner."""
        return polygons_2_bytes(self.polygons.polygons)

    def encode_simplified_polygons(self):
        """Encode the 'simplified_polygons' in the Polygons to space-effecient bytes"""
        return polygons_2_bytes(self.polygons.simplified_polygons)