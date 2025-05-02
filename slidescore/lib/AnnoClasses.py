from collections.abc import Sequence
from typing import Dict, List, Union
import array

from slidescore.lib.simplify import simplifyPolygons
# import numpy as np

class Points(Sequence):
    """Class that allows to store many points space-effeciently. Used to store a mask
    
    Can be indexed to a get a tuple of the n'th point."""
    flattened_points = None
    name = "points"
    metadata = {} # metadata per point "x,y", like: {"21,53": {"type": "lymphocyte"}, "3,4": {"type": "dendrophyl"}}

    def __init__(self, init_points: list = None):
        self.flattened_points = array.array('I')
        super().__init__()

        if init_points:
            for point in init_points:
                self.addPoint(point[0], point[1])

    def __getitem__(self, i: int):
        x = self.flattened_points[i * 2]
        y = self.flattened_points[(i * 2) + 1]
        return (x, y)

    def addPoint(self, x: int, y: int):
        self.flattened_points.extend([x, y])

    def __len__(self):
        return int(len(self.flattened_points) / 2)

class Polygons(Sequence):
    """Somewhat space effecient method of storing the positive and negative vertices from a polygon.
    
    Internally uses EfficientArray to store the positive vertices of each polygon"""
    polygons = None
    simplified_polygons = []
    negative_polygons_i = {}
    labels = []
    name = "polygons"
    metadata = {} # metadata per polygon index, like: {0: {"type": "lymphocyte"}, 3: {"type": "dendrophyl"}}

    def __init__(self):
        self.polygons = EfficientArray()
        self.negative_polygons_i = {}
        self.labels = []
        super().__init__()

    def __getitem__(self, i: int):
        """Retrieves a polygon from the values array and any associated negative polygons, if they are
        associated."""
        points_flat = self.polygons.getValues(i)
        postive_vertices = [(points_flat[i], points_flat[i + 1]) for i in range(0, len(points_flat), 2)]
        return {
            "positiveVertices": postive_vertices,
            "negativeVerticesArr": self.negative_polygons_i[i] if i in self.negative_polygons_i else None
        }

    def addPolygon(self, postive_vertices):
        """Add a polygon to the internal values array and return the index it was assigned"""
        self.polygons.addValues(postive_vertices)
        return len(self.polygons) - 1

    def linkPosPolygonToNegPolygon(self, pos_polygon_i, neg_polygon_i):
        """Store a connection between a positive polygon and a negative polygon, using indices"""
        if pos_polygon_i not in self.negative_polygons_i:
            self.negative_polygons_i[pos_polygon_i] = []
        self.negative_polygons_i[pos_polygon_i].append(neg_polygon_i)

    def simplify(self):
        """Simplifies the stored polygons to 1 px accuracy, and stores further simplified polygons lookup tables"""
        self.polygons = simplifyPolygons(self.polygons, 1)
        # Hard coded tolerance of 16 pixels for now
        self.simplified_polygons = simplifyPolygons(self.polygons, 16)

    def __len__(self):
        return len(self.polygons) # Number of polygons present, pos & neg

class Heatmap():
    """Stores an x/y/value map of a heatmap"""
    matrix: list
    x_offset: int
    y_offset: int
    size_per_pixel: int
    name = "heatmap"

    def __init__(self, data: list, x_offset: int, y_offset: int, size_per_pixel: int):
        # data is 2d matrix containing the pixels
        self.matrix = self.generate_2d_ubyte_array(len(data), len(data[0]))
        self.copy_matrix_to_larger(data, self.matrix)

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.size_per_pixel = size_per_pixel

        super().__init__()

    def setPoint(self, x: int, y: int, value: int):
        """Sets a point in the heatmap, increases matrix array if needed"""
        # First check if the current matrix can hold this xy
        current_size = len(self.matrix), len(self.matrix[0])
        # Matrix is indexed with [y][x]!
        max_y = max(current_size[0], y + 1)
        max_x = max(current_size[1], x + 1)
        
        if max_y > current_size[0] or max_x > current_size[1]:
            new_matrix = self.generate_2d_ubyte_array(max_y, max_x)
            self.copy_matrix_to_larger(self.matrix, new_matrix)
            self.matrix = new_matrix

        # Then simply assign the value
        self.matrix[y][x] = value

    def get_metadata(self):
        metadata = {}
        metadata['x'] = self.x_offset
        metadata['y'] = self.y_offset
        metadata['sizePerPixel'] = self.size_per_pixel
        return metadata

    def generate_2d_ubyte_array(self, num_rows, num_cols):
        """Generate a 2D array of type 'ubyte' with the specified number of rows and columns.
        Returns:
        - A 2D list of `array.array` of type 'B' (unsigned byte).
        """
        return [array.array('B', [0] * num_cols) for _ in range(num_rows)]

    def copy_matrix_to_larger(self, source, target):
        """Copies a smaller matrix into a larger one.
    
        The function assumes that the target matrix is large enough to contain the source matrix.
        """
        for i in range(len(source)):
            for j in range(len(source[0])):
                target[i][j] = source[i][j]

    def __len__(self):
        return len(self.matrix) * len(self.matrix[0]) # Number of bytes occupied

class EfficientArray():
    """Efficient way to represent a array of arrays containing only unsigned integers"""
    valuesArray = None # array.array('I')
    offsetArray = None # array.array('I')
    curOffsetIndex = None # 0

    def __init__(self):
        self.offsetArray = array.array('I')
        self.valuesArray = array.array('I')
        self.curOffsetIndex = 0

        self.offsetArray.append(0)

    def addValues(self, values):
        """Add a list of numbers, uint16t by default, to the current values array"""
        offset = self.offsetArray[self.curOffsetIndex]
        # Add the values and store the new offset

        self.valuesArray.extend(values)
        self.curOffsetIndex += 1
        self.offsetArray.append(offset + len(values))

    def getValues(self, i: int):
        """Retrieve an entry from the values array"""
        if i >= self.curOffsetIndex:
            print("Trying to get i", i, "but max is", self.curOffsetIndex)
            return None
        start = self.offsetArray[i]
        end = self.offsetArray[i + 1]
        return self.valuesArray[start : end]

    def __len__(self):
        return self.curOffsetIndex

# Types
Items = Union[Points, Polygons, Heatmap]

# Single item
Point = List[int] # Of len == 2
Polygon = Dict[str, Points] # With str == "positiveVertices" | "negativeVertices"
Item = Union[Point, Polygon]

