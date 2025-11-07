# Std libs
import sys
import array
import time
import json
import math
import logging

# Local libs
from .AnnoClasses import EfficientArray, Points, Polygons, Heatmap
from .PolygonContainer import PolygonContainer

# Importing functions
def read_tsv(path: str, points_type: str, support_experimental = False):
    """Function to parse either points or polygons from a .tsv file on disk. Reads the first line to determine
    whether points or polygons are encoded. See their respective functions for the expected format on disk."""
    items = None

    with open(path, 'r') as fh:
        # Read the first line to determine if we are dealing with points or polygons
        first_line = fh.readline()
        line_parts = first_line.split()

    are_points = len(line_parts) == 2
    is_heatmap = line_parts[0].lower() == 'heatmap'
    is_binary_heatmap = line_parts[0].lower() == 'binary-heatmap'
    if is_binary_heatmap and not support_experimental:
        raise Exception('Wanted to encode a binary heatmap but --experimental is not present')

    if is_heatmap:
        items = read_tsv_heatmap(path)
    elif is_binary_heatmap:
        items = read_tsv_binary_heatmap(path)
    elif are_points:
        items = read_tsv_points(path)
        if points_type == "mask":
            items.name = "mask"
    else:
        items = read_tsv_polygons(path)
        
    if len(items) == 0:
        sys.exit("No items loaded")

    return items

def read_tsv_points(path: str):
    """Read lines from a file to extract points. One point, consisting of 2 coordinates seperated by a tab, should be encoded per line.
    Like this:
    ```
    x1 y1  
    x2 y2  
    etc.
    ```
    """
    items = Points()

    with open(path, 'r') as fh:
        for raw_line in fh:
            line_parts = raw_line.strip().split()
            if len(line_parts) < 2:
                continue

            x, y = (int(line_parts[0]), int(line_parts[1]))
            items.addPoint(x, y)

    return items

def read_tsv_polygons(path: str):
    """Read lines from a file to extract polygons. One polygon is encoded per line, with all the coordinates
    seperated by tab (or any whitespace), like this: `x1  y1  x2 y2  etc.`
    """
    items = Polygons()

    with open(path, 'r') as fh:
        for raw_line in fh:
            
            line_parts = raw_line.strip().split()
            if len(line_parts) < 2:
                continue

            cur_polygon = [int(point) for point in line_parts]
            items.addPolygon(cur_polygon)

    return items

def read_geo_json(path: str):
    """Assumed are the QuPath GeoJSON files, containing only polygons or points."""
    # Return either of these, show warning if both have entries
    polygons = Polygons()
    points = Points()

    with open(path, 'r') as fh:
        data = json.load(fh)
    
    polygon_or_points_generator = extract_geojson(data)
    for metadata, positive_vertices, negative_vertices_list in polygon_or_points_generator:
        if len(positive_vertices) == 1:
            # This is a point, not a polygon
            point = positive_vertices[0]
            x, y = round(point[0]), round(point[1])
            points.addPoint(x, y)

            if metadata is not None:
                points.metadata[f'{x},{y}'] = metadata
        else:
            # Round the positive vertices into ints and add them to a Polygons class
            cur_polygon = []
            for point in positive_vertices:
                x, y = round(point[0]), round(point[1])
                cur_polygon.extend([x, y])
            pos_polygon_i = polygons.addPolygon(cur_polygon)

            # Add per-polygon metadata
            if metadata is not None:
                polygons.metadata[pos_polygon_i] = metadata

            # Add any negative polygons, if present
            for negative_vertices in negative_vertices_list:
                cur_neg_polygon = []
                for point in negative_vertices:
                    x, y = round(point[0]), round(point[1])
                    cur_neg_polygon.extend([x, y])
                neg_polygon_i = polygons.addPolygon(cur_neg_polygon)

                polygons.linkPosPolygonToNegPolygon(pos_polygon_i, neg_polygon_i)

    if len(points) == 0 and len(polygons) == 0:
        raise Exception("No points or polygons loaded from GeoJSON")

    if len(points) != 0 and len(polygons) == 0:
        return points

    if len(points) == 0 and len(polygons) != 0:
        return polygons

    # We got _both_ polygons and points
    print("WARNING: Detected BOTH points and polygons in GeoJSON, only continueing with polygons", file=sys.stderr)
    print("WARNING: Please remove the points from the GeoJSON to prevent ambiguity", file=sys.stderr)
    return polygons

def extract_geojson(data):
    """Yields all polygons and points and their negative vertices from a QuPath GeoJSON data object like so:
    yield (metadata, positiveVertices, [negativeVertices1, negativeVertices2, ...])
    if postiveVertices is of length 2, store them as points
    https://datatracker.ietf.org/doc/html/rfc7946
    """
    for feature in data["features"]:
        if "geometry" not in feature:
            continue
        metadata = feature.get('properties')

        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            # The first entry is the "exterior" ring and the others "interior" rings
            # so the second ones are deemed "negative" polygons, because they are holes
            yield (metadata, geometry["coordinates"][0], geometry["coordinates"][1:])
            
            if "nucleusGeometry" not in feature:
                continue
            nucl_geometry = feature["nucleusGeometry"]
            if nucl_geometry["type"] != "Polygon":
                continue
            yield (metadata, nucl_geometry["coordinates"][0], nucl_geometry["coordinates"][1:])
        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                yield (None, polygon[0], polygon[1:])
        elif geometry["type"] == "Point":
            # If we got a GeoJSON with points, report them as a polygon of length 2
            # Callee should handle this
            for polygon in geometry["coordinates"]:
                yield (metadata, [geometry["coordinates"]], [])

def read_slidescore_json(data):
    """Parses JSON's that are created using the SlideScore Front-end.
    Currently the supported formats are: points, polygons and brush entries.
    If an unsupported type is encoutered, an error is thrown.

    data: List of points, polygons or brush entries

    WARNING: All information except coordinates is lost, like ["label"] information
    """
    if not isinstance(data, list):
        raise Exception("Expected a list as data")

    if len(data) == 0:
        raise Exception("Data is an empty list, cannot convert")

    # First check if data is points, they are stored as a raw list of xy pairs
    if "type" not in data[0]: 
        if 'x' in data[0] and 'y' in data[0]:
            items = Points()
            items.name = "points" # Define that it is not a mask, but circles
            for point in data:
                x = int(point["x"])
                y = int(point["y"])
                items.addPoint(x, y)
            return items
        else:
            raise Exception(f'Unsupported slidescore JSON: type not specified')

    # Then check if heatmap or brush / polygon    
    if data[0]["type"] == "heatmap":
        x_offset = data[0]["x"] if 'x' in data[0] else 0
        y_offset = data[0]["y"] if 'y' in data[0] else 0

        items = Heatmap(
            data = data[0]["data"], 
            x_offset = x_offset, 
            y_offset = y_offset,
            # Divide total height by num pixels in column to get height per pixel
            size_per_pixel = round(data[0]["height"] / len(data[0]["data"]))
        )
        return items

    items = Polygons()
    # Either polygon or brush entries, both are polygons
    for entry in data:
        if entry["type"].lower() == 'polygon':
            # Process polygon points
            cur_polygon = []
            for point in entry["points"]:
                x, y = round(point["x"]), round(point["y"])
                cur_polygon.extend([x, y])
            polygon_i = items.addPolygon(cur_polygon)
            # Add any labels if present
            if 'labels' in entry:
                for label in entry['labels']:
                    label['polygon_i'] = polygon_i
                    items.labels.append(label)
        elif entry["type"].lower() == 'brush':
            # Process brush polygon, positive polygons first
            pos_polygon_is = [] # Store the indices of all the postive polygons
            for polygon in entry["positivePolygons"]:
                cur_polygon = []
                for point in polygon:
                    x, y = round(point["x"]), round(point["y"])
                    cur_polygon.extend([x, y])
                pos_polygon_i = items.addPolygon(cur_polygon)
                pos_polygon_is.append(pos_polygon_i)
            # Process any negative polygons
            for neg_polygon in entry["negativePolygons"]:
                # Parse the negative polygon
                cur_neg_polygon = []
                for point in neg_polygon:
                    x, y = round(point["x"]), round(point["y"])
                    cur_neg_polygon.extend([x, y])
                neg_polygon_i = items.addPolygon(cur_neg_polygon)
                # Link them to all the positive polygons in this entry
                for pos_polygon_i in pos_polygon_is:
                    items.linkPosPolygonToNegPolygon(pos_polygon_i, neg_polygon_i)
            # Add any labels if present
            if 'labels' in entry:
                for label in entry['labels']:
                    label['polygon_i'] = pos_polygon_is[0]
                    items.labels.append(label)
        elif entry["type"].lower() == 'ellipse':      
            #Turn ellipse into 40 point polygon
            #adapted from https://stackoverflow.com/questions/22694850/approximating-an-ellipse-with-a-polygon
            center = entry["center"]
            size = entry["size"]
            retq1 = []
            retq2 = []
            retq3 = []
            retq4 = []
            n=10

            for i in range(n):
                theta = math.pi / 2 * i / n
                fi = math.pi - math.atan(math.tan(theta) * math.sqrt(size['x'] / size['y']))
                cos = size['x'] * math.cos(fi)
                sin = size['y'] * math.sin(fi)

                x = round(center['x'] + cos)
                y = round(center['y'] + sin)
                retq1.append([x, y])

                x = round(center['x'] - cos)
                y = round(center['y'] + sin)
                retq2.append([x, y])

                x = round(center['x'] - cos)
                y = round(center['y'] - sin)
                retq3.append([x, y])

                x = round(center['x'] + cos)
                y = round(center['y'] - sin)
                retq4.append([x, y])

            retq2.reverse()
            retq4.reverse()
            cur_polygon = []
            for p in retq1:
                cur_polygon.extend([p[0], p[1]])
            for p in retq2:
                cur_polygon.extend([p[0], p[1]])
            for p in retq3:
                cur_polygon.extend([p[0], p[1]])
            for p in retq4:
                cur_polygon.extend([p[0], p[1]])

            polygon_i = items.addPolygon(cur_polygon)
            # Add any labels if present
            if 'labels' in entry:
                for label in entry['labels']:
                    label['polygon_i'] = polygon_i
                    items.labels.append(label)
        elif entry["type"].lower() == 'rect':      
            #Turn rect into polygon
            #adapted from https://stackoverflow.com/questions/22694850/approximating-an-ellipse-with-a-polygon
            corner = entry["corner"]
            size = entry["size"]
            cur_polygon = [round(corner["x"]), round(corner["y"]),
                round(corner["x"]+size["x"]), round(corner["y"]),
                round(corner["x"]+size["x"]), round(corner["y"]+size["y"]),
                round(corner["x"]), round(corner["y"]+size["y"])]
            
            polygon_i = items.addPolygon(cur_polygon)
            # Add any labels if present
            if 'labels' in entry:
                for label in entry['labels']:
                    label['polygon_i'] = polygon_i
                    items.labels.append(label)                    

        else:
            raise Exception(f'Unsupported slidescore JSON type: "{entry["type"]}" not supported')
    return items

def read_tsv_heatmap(path: str):
    """Read lines from a file to extract heatmap points. One point, consisting of 2 coordinates along with a value seperated by a tab, should be encoded per line.
    The first line should be a header, with the first word being "Heatmap", then the x and y offset, and last the size per pixel
    Like this:
    ```
    Heatmap 100 100 16 # x_offset y_offset size_per_pixel
    x1 y1 value1
    x2 y2 value2
    etc.
    ```
    """
    with open(path, 'r') as fh:
        first_line_parts = fh.readline().split()
        x_offset = int(first_line_parts[1])
        y_offset = int(first_line_parts[2])
        size_per_pixel = int(first_line_parts[3])

        # First determine the heatmap size to prevent unneeded copies
        prev_poss = fh.tell()
        max_y = 0
        max_x = 0
        for line in fh:
            line_parts = line.split()
            x, y, value = int(line_parts[0]), int(line_parts[1]), int(line_parts[2])
            max_y = max(max_y, y + 1)
            max_x = max(max_x, x + 1)

        fh.seek(prev_poss) # go back to beginning of data

        # Construct the heatmap
        data = [[0] * max_x for _ in range(max_y)]
        # Parse again to save the results
        heatmap = Heatmap(data, x_offset, y_offset, size_per_pixel)
        for line in fh:
            line_parts = line.split()
            x, y, value = int(line_parts[0]), int(line_parts[1]), int(line_parts[2])
            heatmap.setPoint(x, y, value)

    return heatmap

def read_tsv_binary_heatmap(path: str):
    """Read lines from a file to extract binary heatmap points. One point, consisting of 2 coordinates seperated by a tab, should be encoded per line.
    The first line should be a header, with the first word being "binary-heatmap", then the x and y offset, and last the size per pixel
    Like this:
    ```
    binary-heatmap 100 100 16 # x_offset y_offset size_per_pixel
    x1 y1
    x2 y2
    etc.
    ```
    """
    with open(path, 'r') as fh:
        first_line_parts = fh.readline().split()
        x_offset = int(first_line_parts[1])
        y_offset = int(first_line_parts[2])
        size_per_pixel = int(first_line_parts[3])

        # First determine the heatmap size to prevent unneeded copies
        prev_poss = fh.tell()
        max_y = 0
        max_x = 0
        for line in fh:
            line_parts = line.split()
            x, y = int(line_parts[0]), int(line_parts[1])
            max_y = max(max_y, y + 1)
            max_x = max(max_x, x + 1)

        fh.seek(prev_poss) # go back to beginning of data

        # Construct the heatmap
        data = [[0] * max_x for _ in range(max_y)]
        # Parse again to save the results


        # Construct the heatmap
        heatmap = Heatmap(data, x_offset, y_offset, size_per_pixel)
        for line in fh:
            line_parts = line.split()
            x, y = int(line_parts[0]), int(line_parts[1])
            heatmap.setPoint(x, y, 255)
    heatmap.name = 'binary-heatmap'
    return heatmap

# Export functions
supported_types = {
    'B': 'uint8',
    'H': 'uint16',
    'I': 'uint32'
}

def encode_typed_arr(obj):
    """Encodes a array.array into a bytebuffer and container object"""
    if len(obj) == 0:
        return []
    if obj.typecode not in supported_types:
        raise Exception('Unsupported typed array')

    array_type = supported_types[obj.typecode]

    typed_array_obj = {
        "isTypedArray": True,
        "bytes": obj.tobytes(),
        "type": array_type,
        "len": len(obj)
    }
    return typed_array_obj

def encode_effecient_arr(obj: EfficientArray):
    """Encodes an EfficientArray into a container object with a destructered representation"""
    return {
        "isEfficientArray": True,
        "data": {
            "offsetArray": obj.offsetArray,
            "valuesArray": obj.valuesArray,
            "length": len(obj)
        }
    }

def encode_polygon_container(obj: PolygonContainer):
    """Encodes a polygon container into a space effecient polygons buffer and the tile and negative polygons information"""
    return {
        "isPolygonContainer": True,
        "data": {
            "allTiles": obj.allTiles,
            "polygons": obj.encode_polygons(), # encode_effecient_arr(obj.polygons),
            "negativePolygons": obj.polygons.negative_polygons_i,
            "tileSize": obj.tile_size
        }
    }

def msgpack_encoder(obj):
    """Encoder that calls encode_polygon_container & encode_typed_arr for their respective objects"""
    if isinstance(obj, PolygonContainer):
        return encode_polygon_container(obj)

    if isinstance(obj, array.array):
        return encode_typed_arr(obj)

# miscellaneous

time_on_load = time.time()
def log(*args):
    """Logs the arguments to the console, prefixing the time passed since script execution began"""
    time_passed = time.time() - time_on_load
    print("{:.2f}".format(time_passed), ' '.join(map(str, args)))


NOTICE_LEVEL = 25  # between INFO (20) and WARNING (30)
logging.addLevelName(NOTICE_LEVEL, "NOTICE")

def notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE_LEVEL):
        self._log(NOTICE_LEVEL, message, args, **kwargs)

logging.Logger.notice = notice

def get_logger(verbosity: int) -> logging.Logger:
    """Configure and return a logger with the given verbosity level."""
    # Map verbosity count to logging levels
    if verbosity == 0:
        level = NOTICE_LEVEL
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)
    logger.level = level

    return logger
