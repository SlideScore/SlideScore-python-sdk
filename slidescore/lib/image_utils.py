import io 
import png

def gen_matrix(num_row, num_col):
    """Generate a python based matrix with a number of columns and rows."""
    matrix = []
    for _ in range(num_row):
        column = bytearray(num_col)
        matrix.append(column)
    return matrix

def encode_png(matrix, width, height, bitdepth=1):
    """Encode a matrix of pixel values into PNG with the best compression."""
    writer = png.Writer(width, height, greyscale=True, bitdepth=bitdepth, compression=9)
    f = io.BytesIO()
    writer.write(f, matrix)
    f.seek(0)
    png_buf = f.read()
    return png_buf

def get_point_matrix(pointsArr, tile_size):
    """Turns a list of points into a matrix that has 0 for empty pixels and 1 for filled pixels."""
    matrix = gen_matrix(tile_size, tile_size)
    for i in range(0, len(pointsArr), 2):
        x = pointsArr[i]
        y = pointsArr[i + 1]
        matrix[y][x] = 1
    return matrix


def get_png_bytes(pointsArr, tile_size):
    """Encodes a list of points that fall within tile_size x tile_size pixels into a PNG."""
    matrix = get_point_matrix(pointsArr, tile_size)
    return encode_png(matrix, tile_size, tile_size)

def lookup_table_2_png(lookup_table_container):
    """Encodes a lookup table/density map into a greyscale PNG."""
    lookup_table = lookup_table_container["lookup"]

    max_x, max_y, max_val = get_max_vals(lookup_table)
    width, height = max_x + 1, max_y + 1

    matrix = gen_matrix(height, width)

    for y in lookup_table:
        for x in lookup_table[y]:
            new_value = round((lookup_table[y][x] / max_val) * 255)
            matrix[y][x] = new_value

    return encode_png(matrix, width, height, 8)


def get_max_vals(lookup_table):
    """Calculates the width and height of the lookup table and the max value present."""
    max_y = 0
    max_x = 0
    max_val = 0

    for y in lookup_table:
        max_y = max(max_y, y)
        for x in lookup_table[y]:
            max_x = max(max_x, x)
            max_val = max(max_val, lookup_table[y][x])
    return max_x, max_y, max_val