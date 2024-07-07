"""
This test checks the ability to create a study and request image tiles from the server
"""

import sys
import os
from datetime import datetime
import time
import math
import warnings
import sys

import requests # Should be present since it is required by slidescore sdk

import slidescore
from common_lib import create_study

def fetch_img_tile(host: str, image_id: int, img_auth, level: int, col: int, row:int):
    """Fetches an image tile from a SlideScore server at a specified level"""
    url = f'{host}/i/{image_id}/{img_auth["urlPart"]}/i_files/{level}/{col}_{row}.jpeg'
    img_req = requests.get(url,
        cookies={ "t": img_auth["cookiePart"] }
    )
    return img_req.content, img_req

def test_high_perf_img_server():

    SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    USER_EMAIL = os.getenv('SLIDESCORE_EMAIL') or "pytest@example.com"

    # Make sure we got a HOST and KEY
    assert SLIDESCORE_HOST
    assert SLIDESCORE_API_KEY
    assert USER_EMAIL
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)

    # Create study with slide
    datetime_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    study_name = f'test-study-image-{datetime_str}'
    study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL)

    # Fetch image metadata and authentication details for the high performance image server
    img_metadata = client.perform_request("GetImageMetadata", {"imageid": image_id}, method="GET").json()["metadata"]
    assert 'level0TileWidth' in img_metadata
    assert 'level0TileHeight' in img_metadata
    assert 'mppX' in img_metadata
    assert 'levelCount' in img_metadata
    assert 'levelHeights' in img_metadata

    img_auth = client.perform_request("GetTileServer", {"imageid": image_id}, method="GET").json()
    assert 'cookiePart' in img_auth
    assert 'urlPart' in img_auth
    assert 'expiresOn' in img_auth
        
    # Parse the img metadata and request tiles for a zoomed out level
    img_width, img_height = img_metadata["level0Width"], img_metadata["level0Height"]

    tile_size = img_metadata["osdTileSize"]
    max_level = math.ceil(math.log2(max(img_width, img_height)))
    num_levels = max_level + 1
    level = max_level

    # Calculate how many rows and columns there are of this zoom level
    num_tile_columns = math.ceil((img_width / tile_size) / 2 ** (max_level - level))
    num_tile_rows = math.ceil((img_height / tile_size) / 2 ** (max_level - level))
    assert num_tile_columns > 1
    assert num_tile_rows > 1
    
    # Get the most top left tile, and make sure we do it in less than 100ms
    t0 = time.time()
    first_tile_jpeg_bytes, _ = fetch_img_tile(SLIDESCORE_HOST, image_id, img_auth, level, 0, 0)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    assert first_tile_jpeg_bytes.startswith(b'\xff\xd8\xff') # JPEG header
    if dt > 20:
        warnings.warn(f"Image request took over 20 ms, it took {round(dt)} ms.")
    assert dt < 100

    # Get the most bottom right tile
    last_tile_jpeg_bytes, _ = fetch_img_tile(SLIDESCORE_HOST, image_id, img_auth, level, num_tile_columns - 1, num_tile_rows - 1)
    assert last_tile_jpeg_bytes.startswith(b'\xff\xd8\xff')
    
    # Now make sure we cannot fetch a non-existant tile
    jpeg_bytes, req = fetch_img_tile(SLIDESCORE_HOST, image_id, img_auth, level, num_tile_columns, num_tile_rows)
    assert req.status_code != 200
    
    path = client.get_slide_path(image_id)
    assert path
    
    client.update_slide_path(image_id, path.replace("images/","").replace("images\\","").replace("\\","/"))

    localfile = sys.path[0].replace("\\","/")+ '/test-image.TiFF'
    ret = client.perform_request("GenerateSlideFileURL", {
         "filename": localfile,
         "user": USER_EMAIL
         }, method="POST")
         
    assert ret
    rjson = ret.json()
    assert rjson["success"]
    
    assert rjson["link"]
    req = requests.get(SLIDESCORE_HOST + rjson["link"],
        cookies={ "t": img_auth["cookiePart"] })
    assert req
    assert req.content
    assert req.text.find("openseadragon") != -1
    
    ret = client.perform_request("GenerateSlideFileURL", {
         "filename": localfile,
         "user": None
         }, method="POST")
    
    assert ret
    rjson = ret.json()
    assert rjson["success"]
    
    assert rjson["link"]
    req = requests.get(SLIDESCORE_HOST + rjson["link"])
    assert req
    assert req.content
    assert req.text.find("openseadragon") != -1
    

def test_get_raw_tile():
    SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    USER_EMAIL = os.getenv('SLIDESCORE_EMAIL') or "pytest@example.com"

    # Make sure we got a HOST and KEY
    assert SLIDESCORE_HOST
    assert SLIDESCORE_API_KEY
    assert USER_EMAIL
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)

    # Create study with slide
    datetime_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    study_name = f'test-study-image-raw-{datetime_str}'
    study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL)
    raw_tile = client.get_raw_tile(study_id, image_id, 0, 50, 50, 250, 250, 9)
    assert raw_tile.status_code == 200
    assert b'JFIF' in raw_tile.content

if __name__ == "__main__":
    test_get_raw_tile()
    # sys.exit('This file is meant to be ran by PyTest')