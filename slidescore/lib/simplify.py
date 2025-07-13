import array


def getSquareSegmentDistance(p, p1, p2):
    """
    Square distance between point and a segment
    """
    x = p1[0]
    y = p1[1]

    dx = p2[0] - x
    dy = p2[1] - y

    if dx != 0 or dy != 0:
        t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy)

        if t > 1:
            x = p2[0]
            y = p2[1]
        elif t > 0:
            x += dx * t
            y += dy * t

    dx = p[0] - x
    dy = p[1] - y

    return dx * dx + dy * dy

def simplifyDouglasPeucker(points: array.array, tolerance: float):
    length = int(len(points) // 2)
    markers = []

    first = 0
    last = length - 1

    first_stack = []
    last_stack = []

    new_points = array.array('I')

    markers = [first, last]

    while last:
        max_sqdist = 0

        for i in range(first, last):
            point_i = points[i * 2: i * 2 + 2]
            point_first = points[first * 2: first * 2 + 2]
            point_last =points[last * 2: last * 2 + 2]
            sqdist = getSquareSegmentDistance(point_i, point_first, point_last)

            if sqdist > max_sqdist:
                index = i
                max_sqdist = sqdist

        if max_sqdist > tolerance:
            markers.append(index)

            first_stack.append(first)
            last_stack.append(index)

            first_stack.append(index)
            last_stack.append(last)

        # Can pop an empty array in Javascript, but not Python, so check
        # the length of the list first
        if len(first_stack) == 0:
            first = None
        else:
            first = first_stack.pop()

        if len(last_stack) == 0:
            last = None
        else:
            last = last_stack.pop()

    markers.sort()
    for i in markers:
        new_points.extend(points[i * 2: i * 2 + 2])
    
    # [x1, y1, x1, y1] -> [x1, y1]
    if len(new_points) == 4 and new_points[0] == new_points[2] and new_points[1] == new_points[3]:
        new_points.pop()
        new_points.pop()
    return new_points


def simplify(points: array.array, tolerance=1.0):
    sqtolerance = tolerance * tolerance

    points = simplifyDouglasPeucker(points, sqtolerance)

    return points

def gen_chunks(flat_list: list, chunk_size: int):   
    for i in range(0, len(flat_list), chunk_size):
        yield flat_list[i:i + chunk_size]

def simplifyPolygons(polygons_arr, tolerance=1.0):
    from .AnnoClasses import EfficientArray
    simp_polygons_arr = EfficientArray()
    for polygon_i in range(len(polygons_arr)):
        # List of [x1, y1, x2, y2, etc...]
        polygon_orig = polygons_arr.getValues(polygon_i)
        polygon_simp = simplify(polygon_orig, tolerance)
        simp_polygons_arr.addValues(polygon_simp)
    assert len(polygons_arr) == len(simp_polygons_arr)
    return simp_polygons_arr

if __name__ == "__main__":
    from .AnnoClasses import Polygons
    print("Testing simplify algorithm")
    almost_triangle = [0, 0, 100, 100, 105, 95, 200, 0]
    polygons = Polygons()
    polygons.addPolygon(almost_triangle)
    res = simplify(almost_triangle, 1)
    print(list(res))