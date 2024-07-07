from collections.abc import Sequence
from typing import Dict, List, Union
import array
import numpy as np

class Points(Sequence):
    """Class that allows to store many points space-effeciently. Used to store a mask
    
    Can be indexed to a get a tuple of the n'th point."""
    flattened_points = None
    name = "points"

    def __init__(self):
        self.flattened_points = array.array('I')
        super().__init__()

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
    
    Internally uses Points to store the positive vertices of each polygon"""
    polygons = None
    negative_polygons_i = {}
    labels = []
    name = "polygons"

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

    def __len__(self):
        return len(self.polygons) # Number of polygons present, pos & neg

class Heatmap():
    """Stores an x/y/value map of a heatmap"""
    matrix: np.ndarray
    x_offset: int
    y_offset: int
    size_per_pixel: int
    name = "heatmap"

    def __init__(self, data: list, x_offset: int, y_offset: int, size_per_pixel: int):
        self.matrix = np.array(data, dtype=np.ubyte)

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.size_per_pixel = size_per_pixel

        super().__init__()

    def setPoint(self, x: int, y: int, value: int):
        """Sets a point in the heatmap, increases matrix array if needed"""
        # First check if the current matrix can hold this xy
        current_size = self.matrix.shape
        # Matrix is indexed with [y][x]!
        max_y = max(current_size[0], y + 1)
        max_x = max(current_size[1], x + 1)
        
        if max_y > current_size[0] or max_x > current_size[1]:
            new_size = (max_y, max_x)
            new_matrix = np.zeros(new_size, dtype=np.ubyte)
            new_matrix[:current_size[0], :current_size[1]] = self.matrix
            self.matrix = new_matrix

        # Then simply assign the value
        self.matrix[y][x] = value

    def get_metadata(self):
        metadata = {}
        metadata['x'] = self.x_offset
        metadata['y'] = self.y_offset
        metadata['sizePerPixel'] = self.size_per_pixel
        return metadata

    def __len__(self):
        return self.matrix.shape[0] * self.matrix.shape[1] # Number of bytes occupied

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

