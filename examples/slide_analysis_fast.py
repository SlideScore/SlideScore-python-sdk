"""
This example showcases a local image job in a high performance manner.
It uses SlideScore's image server to quickly fetch a tile of a big slide,
this tile then locally gets processed using the OpenCV 2 library. The result
of this operation is then converted into SlideScore's high performance 
binary format and uploaded. Please refer to `slide_analysis_simple.py` for
a easier example for a less demanding usecase.

Date: 3-3-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os
import tempfile
import datetime
import json
import math

import slidescore
import slidescore.lib.utils
import requests

import cv2 # $ pip install opencv-python
import numpy as np # $ pip install numpy

from add_anno2 import get_study_and_image

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('What is your Slidescore email: ') # admin@example.com


def fetch_img_tile(host: str, image_id: int, img_auth, level: int, col: int, row:int):
    """Fetches an image tile from a SlideScore server at a specified level"""
    url = f'{host}/i/{image_id}/{img_auth["urlPart"]}/i_files/{level}/{col}_{row}.jpeg'
    img_req = requests.get(url,
        cookies={ "t": img_auth["cookiePart"] }
    )
    return img_req.content

def contours_2_polygons(contours, level: int, max_level: int, tile_size: int, tile_x: int, tile_y: int):
    """Converts OpenCV contours in an image tile into SlideScore polygon objects with the correct coordinates.
    Coordinates are based on max resolution level."""
    x_factor = 2 ** (max_level - level)
    y_factor = x_factor
    x_offset = x_factor * tile_x * tile_size
    y_offset = y_factor * tile_y * tile_size

    polygons = []
    for contour in contours:
        points = []
        for point in contour:
            # The contours are based on a scaled down version of the image
            # so translate these coordinates to coordinates of the original image
            orig_x, orig_y = int(point[0][0]), int(point[0][1])
            points.append({
                "x": int(x_factor * orig_x) + x_offset,
                "y": int(y_factor * orig_y) + y_offset
            })
        polygon = {
            "type":"polygon",
            "points": points
        }
        polygons.append(polygon)
    return polygons

def tresholded_2_points(tresholded_img, level: int, max_level: int, tile_size: int, tile_x: int, tile_y: int):
    """Converts an OpenCV tresholded image into SlideScore point objects.
    Coordinates are based on the maximum resolution level. Multiple points are added
    if not zoomed in all the way."""
    x_factor = 2 ** (max_level - level)
    y_factor = x_factor
    x_offset = x_factor * tile_x * tile_size
    y_offset = y_factor * tile_y * tile_size

    points = []
    for row_i in range(len(tresholded_img)):
        row = tresholded_img[row_i]
        for col_i in range(len(row)):
            pixel = tresholded_img[row_i][col_i]

            if pixel == 0:
                # The pixel in this tile level covers more than 1 pixel in the base-level
                # annotate them all
                x_pixel_offset = round(-1 * 0.5 * x_factor) # Left to right
                y_pixel_offset = round(-1 * 0.5 * y_factor) # Top to bottom
                for dx in range(x_pixel_offset, x_pixel_offset * -1):
                    for dy in range(y_pixel_offset, y_pixel_offset * -1):
                        points.append({
                            "x": int(x_factor * col_i) + x_offset + dx,
                            "y": int(y_factor * row_i) + y_offset + dy
                        })
    return points


if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST or not SLIDESCORE_EMAIL:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST or SLIDESCORE_EMAIL not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    # Asks which study and image to use using a CLI
    study_id, image_id = get_study_and_image(client)

    # Fetch image metadata and authentication details for the high performance image server
    img_metadata = client.perform_request("GetImageMetadata", {"imageid": image_id}, method="GET").json()["metadata"]
    img_auth = client.perform_request("GetTileServer", {"imageid": image_id}, method="GET").json()

    # Parse the img metadata and request tiles for a zoomed out level
    img_width, img_height = img_metadata["level0Width"], img_metadata["level0Height"]

    if img_width < 10000 or img_height < 10000:
        print("WARNING: Image appears to be quite small, continuing...")

    tile_size = img_metadata["osdTileSize"]

    max_level = math.ceil(math.log2(max(img_width, img_height)))
    num_levels = max_level + 1
    print(f"Detected image with {num_levels} levels")
    level = min(max_level, 13) # Use image zoom level 13

    # Calculate how many rows and columns there are of this zoom level
    num_tile_columns = math.ceil((img_width / tile_size) / 2 ** (max_level - level))
    num_tile_rows = math.ceil((img_height / tile_size) / 2 ** (max_level - level))
    print(f"Image has {num_tile_columns} columns and {num_tile_rows} rows at tile level {level}")

    # Extract dark pixels using OpenCV as a local image analysis job
    treshold = 60
    annotation_type = input("Annotate as polygons or points: ") or 'points'

    all_annotations = [] # Stores SlideScore annotations (points/polygons)
    print("Retrieved image metadata, performing local image analysis.")
    
    # Loop over the columns of the tiled image
    for col_i in range(num_tile_columns): # Left to right
        print(f"Processing column: {col_i + 1} / {num_tile_columns}")
        if len(all_annotations) > 10 * 1000 * 1000:
            print("More than 10 million annotations, not processing further columns to prevent OOM")
            break
        # Loop over the individual tiles in this column
        for row_i in range(num_tile_rows): # Top to bottom
            # Retrieve the image and process it using openCV
            jpeg_bytes = fetch_img_tile(SLIDESCORE_HOST, image_id, img_auth, level, col_i, row_i)
            
            # Threshold the image
            jpeg_as_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(jpeg_as_np, flags=1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, treshold, 255, 0)

            # Convert the tresholded image either to SlideScore polygons, or points
            if annotation_type == 'polygons':
                # Contour image, and save the corrected polygon coordinates in all_annotations
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotations_in_tile = contours_2_polygons(contours[1:], level, max_level, tile_size, col_i, row_i)
            else:
                # Save the coordinates of the tresholded pixels as point annotations
                annotations_in_tile = tresholded_2_points(thresh, level, max_level, tile_size, col_i, row_i)
            all_annotations.extend(annotations_in_tile)


    print(f'Done looping over image, extracted {len(all_annotations)} annotations, converting to anno2')

    # Convert annotation into anno2 for high performance viewing
    anno2_path = os.path.join(tempfile.mkdtemp(suffix='_anno2'), 'anno2.zip')
    metadata = {
        "model-version": 0.1,
        "open-cv2-version": cv2.version.opencv_version
    }
    
    # The following call converts the python SlideScore annotations into a high performance, but
    # low size binary format.
    if annotation_type == 'points':
        all_annotations = slidescore.lib.utils.read_slidescore_json(all_annotations)
        all_annotations.name = 'mask'
    client.convert_to_anno2(all_annotations, metadata, anno2_path)
    print("Created anno2 @", anno2_path, "with size:", int(os.path.getsize(anno2_path) / 1024), 'kiB')

    # Add Anno2 in the database, and receive an uploadtoken
    anno2_resp = client.perform_request("CreateAnno2", {
        "imageId": image_id,
        "studyId": study_id,
        "question": "Annotate shape",
        "email": SLIDESCORE_EMAIL
    }, method="POST").json()
    print("Created anno2 entry in SlideScore, uploading annotation")

    client.upload_using_token(anno2_path, anno2_resp["uploadToken"])

    # Log URL you can use to easily view the result
    print(f'Done, view results at: {SLIDESCORE_HOST}/Image/Details?imageId={image_id}&studyId={study_id}')