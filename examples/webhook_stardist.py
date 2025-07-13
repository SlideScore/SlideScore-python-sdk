DESC="""
Example webhook that uses stardist and the high performance Slide Score API to quickly
identify cells.

Date: 2024-08-30
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import traceback
import json
import time
import requests
import math
import os
import tempfile
import argparse
from io import BytesIO

import slidescore

import tensorflow as tf
import skimage
import numpy as np
from PIL import ImageDraw, Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D


# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_he')

def get_polygons(img, x_offset, y_offset):
    """Uses stardist to retrieve a set of nuclei from an image"""
    mean = np.mean(img)
    if mean > 235:
        # This image is so white, we skip if for regular H&E slides
        return []
    img_n = normalize(img, clip=True)
    labels, res_dict = model.predict_instances(img_n)
    polygons = []
    for i in range(len(res_dict['coord'])):
        # You can also add offsets etc to the coords here
        x_coords = res_dict['coord'][i][1] + x_offset
        y_coords = res_dict['coord'][i][0] + y_offset
        prob = res_dict['prob'][i]
        center = res_dict['points'][i]
        center_img = f"{center[1] + x_offset}, {center[0] + y_offset}"
        center_in_tile = f"{center[1]}, {center[0]}"

        vertices_np = np.column_stack((x_coords,y_coords))
        vertices = vertices_np.tolist()
        vertices.append(vertices[0]) # Make sure the last coordinate is the same as the first

        properties = {
            "type": "stardist",
            "prob": "{:.2f}".format(float(prob)),
            "center_img": center_img,
            "center_in_tile": center_in_tile
        }

        polygons.append({
            "vertices": vertices,
            "properties": properties
        })
    return polygons

def get_rois(answers: list):
    roi_json = next((answer["value"] for answer in answers if answer["name"] == "ROI"), None)
    if roi_json is None:
        raise Exception("Failed to find the ROI answer")
    rois = json.loads(roi_json)
    if len(rois) == 0:
        raise Exception("No ROI given")
    return rois

def fetch_img_tile(session: requests.Session, host: str, image_id: int, img_auth, level: int, col: int, row:int):
    """Fetches an image tile from a SlideScore server at a specified level"""
    url = f'{host}/i/{image_id}/{img_auth["urlPart"]}/i_files/{level}/{col}_{row}.jpeg'
    img_req = session.get(url,
        cookies={ "t": img_auth["cookiePart"] }
    )
    return img_req.content

def rois_to_polygons(rois: list):
    polygons = []
    for roi in rois:
        points = [
            { "x": roi["corner"]["x"], "y": roi["corner"]["y"]},
            { "x": roi["corner"]["x"] + roi["size"]["x"], "y": roi["corner"]["y"]},
            { "x": roi["corner"]["x"] + roi["size"]["x"], "y": roi["corner"]["y"] + roi["size"]["y"]},
            { "x": roi["corner"]["x"], "y": roi["corner"]["y"] + roi["size"]["y"]},
        ]
        polygon = {
            "type":"polygon",
            "points": points
        }
        polygons.append(polygon)
    return polygons

def chunk_indices(start_col_i, end_col_i, start_row_i, end_row_i, chunk_size=4):
    # Create lists to hold the chunks
    col_chunks = []
    row_chunks = []

    # Generate column chunks
    for i in range(start_col_i, end_col_i + 1, chunk_size):
        col_chunk = tuple(range(i, min(i + chunk_size, end_col_i + 1)))
        col_chunks.append(col_chunk)

    # Generate row chunks
    for i in range(start_row_i, end_row_i + 1, chunk_size):
        row_chunk = tuple(range(i, min(i + chunk_size, end_row_i + 1)))
        row_chunks.append(row_chunk)

    return col_chunks, row_chunks

def generate_chunks(start_col_i, end_col_i, start_row_i, end_row_i, chunk_size=4):
    col_chunks, row_chunks = chunk_indices(start_col_i, end_col_i, start_row_i, end_row_i, chunk_size)
    
    options = []
    
    # Combine each row chunk with each column chunk
    for row_chunk in row_chunks:
        for col_chunk in col_chunks:
            options.append((row_chunk, col_chunk))
    
    return options


def stitch_tiles_to_mosaics(host, image_id, img_auth, level, start_col_i, end_col_i, start_row_i, end_row_i, tile_size):
    # Calculate the number of tiles needed in each dimension
    num_tiles_x = (end_col_i - start_col_i + 1)
    num_tiles_y = (end_row_i - start_row_i + 1)

    # Create a blank image for the mosaic
    mosaic_width = num_tiles_x * tile_size
    mosaic_height = num_tiles_y * tile_size
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height))

    # Loop over the columns of the tiled image
    session = requests.Session()
    for col_i in range(start_col_i, end_col_i + 1):  # Left to right        
        # Loop over the individual tiles in this column
        for row_i in range(start_row_i, end_row_i + 1):  # Top to bottom
            # Retrieve the image and process it
            jpeg_bytes = fetch_img_tile(session, host, image_id, img_auth, level, col_i, row_i)
            image_stream = BytesIO(jpeg_bytes)

            # Read the image using PIL
            tile_image = Image.open(image_stream)

            # Calculate the position to paste the tile
            x_position = (col_i - start_col_i) * tile_size
            y_position = (row_i - start_row_i) * tile_size

            # Paste the tile into the mosaic
            mosaic.paste(tile_image, (x_position, y_position))
    return mosaic


def handle_post(host: str, api_token: str, image_id: int, roi: list):
    client = slidescore.APIClient(host, api_token)
    # Fetch image metadata and authentication details for the high performance image server
    img_metadata = client.perform_request("GetImageMetadata", {"imageid": image_id}, method="GET").json()["metadata"]
    img_auth = client.perform_request("GetTileServer", {"imageid": image_id}, method="GET").json()

    # Parse the img metadata and request tiles for a zoomed out level
    img_width, img_height = img_metadata["level0Width"], img_metadata["level0Height"]

    if img_width < 10000 or img_height < 10000:
        print("WARNING: Image appears to be quite small, continuing...")
    tile_size = img_metadata["osdTileSize"]
    print("Got image metadata", img_metadata)

    max_level = math.ceil(math.log2(max(img_width, img_height)))
    num_levels = max_level + 1
    print(f"Detected image with {num_levels} levels")
    level = max_level

    # Calculate how many rows and columns there are of this zoom level
    num_tile_columns = math.ceil((img_width / tile_size) / 2 ** (max_level - level))
    num_tile_rows = math.ceil((img_height / tile_size) / 2 ** (max_level - level))
    print(f"Image has {num_tile_columns} columns and {num_tile_rows} rows at tile level {level}")


    all_annotations = [] # Stores SlideScore annotations (points/polygons)
    print("Retrieved image metadata, performing local image analysis.")
    
    start_col_i = int(roi["corner"]["x"] / tile_size)
    end_col_i = start_col_i + int(roi["size"]["x"] / tile_size)

    start_row_i = int(roi["corner"]["y"] / tile_size)
    end_row_i = start_row_i + int(roi["size"]["y"] / tile_size)

    # Created chunks (mosaics) that are processed together, of 4x4 tiles
    mosaic_size = 8
    options = generate_chunks(start_col_i, end_col_i + 1, start_row_i, end_row_i + 1, mosaic_size)

    time_spend_fetching = 0
    time_spend_processing = 0
    for row_is, col_is in options:
        start_col_i_mosaic = col_is[0]
        end_col_i_mosaic   = col_is[-1]
        start_row_i_mosaic = row_is[0]
        end_row_i_mosaic   = row_is[-1]

        print(f"Processing mosiac: cols {col_is} rows {row_is}")
        time_s = time.time()
        mosaic = stitch_tiles_to_mosaics(host, image_id, img_auth, level, start_col_i_mosaic, end_col_i_mosaic, start_row_i_mosaic, end_row_i_mosaic, tile_size)
        time_p = time.time()
        # mosaic.save(f'/tmp/mosaic_{start_col_i_mosaic}_{end_col_i_mosaic}_{start_row_i_mosaic}_{end_row_i_mosaic}.png')
        # Convert the PIL image to a NumPy array
        image_array = np.array(mosaic)

        annotations_in_tile = get_polygons(image_array, start_col_i_mosaic * tile_size, start_row_i_mosaic * tile_size)
        all_annotations.extend(annotations_in_tile)
        time_d = time.time()
        time_spend_fetching   += time_p - time_s
        time_spend_processing += time_d - time_p


    print(f'Done looping over image, extracted {len(all_annotations)} annotations, converting to anno2')

    if len(all_annotations) == 0:
        return {
            "type": "text",
            "name": "Stardist results",
            "value": "No results found",
            "color": "#00FFFF"
        }, time_spend_fetching, time_spend_processing
    # Convert annotation into anno2 for high performance viewing
    anno2_path = os.path.join(tempfile.mkdtemp(suffix='_anno2'), 'anno2.zip')
    metadata = {
        "model-version": 0.1,
        "stardist-version": '?'
    }
    slidescore_polygons = []
    for polygon in all_annotations:
        vertices = polygon['vertices']
        points = []
        for x, y in vertices:
            points.append({
                'x': x,
                'y': y
            })
        slidescore_polygons.append({
            'type': 'polygon',
            'points': points
        })

    client.convert_to_anno2(slidescore_polygons, metadata, anno2_path)
    print("Created anno2 @", anno2_path, "with size:", int(os.path.getsize(anno2_path) / 1024), 'kiB')
    response = client.perform_request("CreateOrphanAnno2", {}, method="POST").json()
    assert response["success"] is True

    client.upload_using_token(anno2_path, response["uploadToken"])
    uuid = response["annoUUID"]

    return {
        "type": "anno2",
        "name": "Stardist results",
        "value": uuid,
        "color": "#00FFFF"
    }, time_spend_fetching, time_spend_processing

def request_handler(request):
    time_got_request = time.time()
    host = request["host"]
    image_id = int(request["imageid"])
    answers = request["answers"] # Answers to the questions field, needs to be validated to contain the expected vals
    apitoken = request["apitoken"] # Api token that is generated on the fly for this request
    rois = get_rois(answers) # Get Regions Of Interest
    if len(rois) != 1:
        raise Exception("Please define a single Region Of Interest")
    stardist_res, time_spend_fetching, time_spend_processing = handle_post(host, apitoken, image_id, rois[0])
    device = 'GPU' if len(tf.config.list_physical_devices('GPU')) > 0 else 'CPU'
    return [
        stardist_res,
        {
            "type": "polygons", 
            "name": "Input ROI's", 
            "value": rois_to_polygons(rois),
            "color": "#FF0000"
        },
        {
            "type": "text",
            "name": "Duration",
            "value": f'These results took {(time.time() - time_got_request):.2f} s to generate ({device}); {time_spend_fetching:.2f} fetching, {time_spend_processing:.2f} processing'
        }
    ]

class ExampleAPIServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes("Hello world", "utf-8"))
    
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        if content_len < 10 or content_len > 4096:
            self.send_response(400)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Invalid request", "utf-8"))
        try:
            post_body = self.rfile.read(content_len).decode()
            request = json.loads(post_body)
            response = request_handler(request)
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            self.wfile.write(bytes(json.dumps(response), "utf-8"))

        except Exception as e:
            print("Caught exception:", e)
            print(traceback.format_exc())

            print(post_body)
            self.send_response(500)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Unknown error: " + str(e), "utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('--host', type=str, default='localhost', help='HOST to listen on')
    parser.add_argument('--port', type=int, default=8000, help='PORT to listen on')
    parser.add_argument('--mode', type=str, default='cli', help='cli or server mode')
    parser.add_argument('--input', type=str, default='input.jpg', help='Input file for cli mode')
    parser.add_argument('--output', type=str, default='output.png', help='output file for cli mode')
    args = parser.parse_args()

    if args.mode == 'server':
        webServer = HTTPServer((args.host, args.port), ExampleAPIServer)
        print(f"Server started http://{args.host}:{args.port}, configure your slidescore instance with a default webhook pointing to this host.")
        try:
            webServer.serve_forever()
        except KeyboardInterrupt:
            pass

        webServer.server_close()
        print("Server stopped.")
    else:
        print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
        screenshot = skimage.io.imread(args.input)
        polygons = get_polygons(screenshot, 0, 0)
        image = Image.fromarray(screenshot, mode='RGB').convert("RGBA")
        
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            polygon_points = [(int(p[0]), int(p[1])) for p in polygon['vertices']]
            # print(polygon_points)
            draw.polygon(polygon_points, fill=None, outline='red')

        # Save or show the image
        image.save(args.output)