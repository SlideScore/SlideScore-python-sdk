DESC = """
Async webhook for multi‑slide analysis

Date: 01-08-2025
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import argparse
import asyncio
import json
import tempfile
import traceback
import time
import uuid
import logging
import base64

from aiohttp import web  # pip install aiohttp
import cv2  # $ pip install opencv-python
import numpy as np  # $ pip install numpy
import slidescore


def create_tmp_file(content: str, suffix='.tmp'):
    """Creates a temporary file, used for intermediate files"""
    fd, name = tempfile.mkstemp(suffix=suffix)
    if content:
        with open(fd, 'w') as fh:
            fh.write(content)
    return name


def convert_2_anno2_uuid(items, client, metadata=''):
    """Convert to anno2 zip, upload, and return uploaded anno2 uuid"""
    local_anno2_path = create_tmp_file('', '.zip')
    client.convert_to_anno2(items, metadata, local_anno2_path)
    response = client.perform_request("CreateOrphanAnno2", {}, method="POST").json()
    assert response["success"] is True

    client.upload_using_token(local_anno2_path, response["uploadToken"])
    return response["annoUUID"]


def convert_contours_2_polygons(contours, cur_img_dims, roi):
    """Converts OpenCV2 contours to AnnoShape Polygons format of SlideScore"""
    x_factor = roi["size"]["x"] / cur_img_dims[0]
    y_factor = roi["size"]["y"] / cur_img_dims[1]
    x_offset = roi["corner"]["x"]
    y_offset = roi["corner"]["y"]

    polygons = []
    for contour in contours:
        points = []
        for point in contour:
            orig_x, orig_y = int(point[0][0]), int(point[0][1])
            points.append({
                "x": x_offset + int(x_factor * orig_x),
                "y": y_offset + int(y_factor * orig_y)
            })
        polygons.append({
            "type": "polygon",
            "points": points
        })
    return polygons


def threshold_image(client, image_id: int, rois: list):
    """Extract pixel information by making a "screenshot" of each region of interest"""
    polygons = []
    for roi in rois:
        if roi["corner"]["x"] is None or roi["corner"]["y"] is None:
            continue  # Basic validation

        image_response = client.perform_request(
            "GetScreenshot",
            {
                "imageid": image_id,
                "x": roi["corner"]["x"],
                "y": roi["corner"]["y"],
                "width": roi["size"]["x"],
                "height": roi["size"]["y"],
                "level": 15,
                "showScalebar": "false"
            },
            method="GET"
        )
        jpeg_bytes = image_response.content
        print("Retrieved image from server, performing analysis using OpenCV")

        treshold = 220
        jpeg_as_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(jpeg_as_np, flags=1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, treshold, 255, 0)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        print("Performed local image analysis")

        cur_img_dims = (img.shape[1], img.shape[0])
        roi_polygons = convert_contours_2_polygons(
            contours, cur_img_dims, roi
        )
        polygons.extend(roi_polygons)
        print("Converted image analysis results to SlideScore annotation")
    return polygons


def get_output_type(answers: list):
    """Extract desired output type from the answers list"""
    print("answers", answers)
    output_json = next((answer["value"] for answer in answers if answer["name"] == "Output type"), None)
    if output_json is None:
        return "polygons"  # default
    return output_json.lower()


def convert_polygons_2_centroids(polygons):
    centroids = []
    for polygon in polygons:
        sum_x = 0
        sum_y = 0
        for point in polygon['points']:
            sum_x += point['x']
            sum_y += point['y']
        centroids.append({
            "x": sum_x / len(polygon['points']),
            "y": sum_y / len(polygon['points']),
        })
    return centroids


def convert_points_2_heatmap(points, size_per_pixel=64):
    """Creates an anno1 heatmap object from a set of points, size_per_pixel is in image pixels per heatmap "pixel" """
    # Figure out the size of the heatmap
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for point in points:
        min_x, max_x = min(min_x, point['x']), max(max_x, point['x'])
        min_y, max_y = min(min_y, point['y']), max(max_y, point['y'])

    # Fill the heatmap data with empty rows
    num_columns = int((max_x - min_x) // size_per_pixel + 1)
    num_rows    = int((max_y - min_y) // size_per_pixel + 1)
    heatmap_data = [ [0] * num_columns for row_i in range(num_rows) ]

    # Populate the heatmap with the points data
    max_heatmap_val = 1
    for point in points:
        heatmap_x = int((point['x'] - min_x) // size_per_pixel)
        heatmap_y = int((point['y'] - min_y) // size_per_pixel)
        heatmap_data[heatmap_y][heatmap_x] += 1
        max_heatmap_val = max(max_heatmap_val, heatmap_data[heatmap_y][heatmap_x])

    # Remap heatmap data to be between 0 and 255
    for heatmap_y in range(num_rows):
        for heatmap_x in range(num_columns):
            heatmap_data[heatmap_y][heatmap_x] = round((heatmap_data[heatmap_y][heatmap_x] / max_heatmap_val) * 255)

    # Return full object
    heatmap = {
        "x": min_x,
        "y": min_y,
        "height": max_y - min_y,
        "data": heatmap_data,
        "type": "heatmap"
    }
    return heatmap


# --------------------------------------------------------------------------- #
# Async server implementation
# --------------------------------------------------------------------------- #

# Global job store - maps job_id to status and optional result
job_store = {}
job_lock = asyncio.Lock()


async def handle_post(request):
    """Handle incoming POST - start background job and return job id."""
    try:
        post_body = await request.text()
        request_json = json.loads(post_body)
        request_json['start_time'] = time.time()
    except Exception as e:
        return web.json_response(
            {"status": "error", "message": f"Invalid JSON: {e}"}, status=400
        )

    job_id = str(uuid.uuid4())
    async with job_lock:
        job_store[job_id] = {"status": "processing"}

    # Start background task
    asyncio.create_task(process_job(job_id, request_json))

    return web.json_response({"status": "processing", "id": job_id})


# --------------------------------------------------------------
# 1.  process_job - run the heavy analysis in the background
# --------------------------------------------------------------
async def process_job(job_id: str, request_json: dict) -> None:
    """
    Background worker that performs the heavy analysis.  The function
    writes the final result into ``job_store`` as a *list* of objects
    in the format expected by the Slides client.

    Parameters
    ----------
    job_id : str
        Unique identifier that the client used when posting the job.
    request_json : dict
        All data that the client supplied in the POST request.
    """
    try:
        # 1.1  Unpack the request ------------------------------------------------
        host          = request_json["host"]
        study_id      = int(request_json["studyid"])
        image_id      = int(request_json["imageid"])
        imagename     = request_json["imagename"]
        case_id       = int(request_json["caseid"])
        email         = request_json["email"]
        analysis_id   = int(request_json["analysisid"])
        analysis_name = request_json["analysisname"]
        case_name     = request_json["casename"]
        answers       = request_json["answers"]
        apitoken      = request_json["apitoken"]

        client = slidescore.APIClient(host, apitoken)

        # 1.2  Retrieve image metadata --------------------------------------------
        img_metadata = (
            client.perform_request("GetImageMetadata",
                                   {"imageid": image_id},
                                   method="GET")
            .json()["metadata"]
        )

        img_width, img_height = img_metadata["level0Width"], img_metadata["level0Height"]
        rois = [{"corner": {"x": 0, "y": 0},
                 "size": {"x": img_width, "y": img_height}}]

        # 1.3  Determine what the user wants --------------------------------------
        output_type = get_output_type(answers)          # "polygons" | "points" | "heatmap"

        # 1.4  Run the heavy analysis (this is the slow part) --------------------
        result_polygons = threshold_image(client, image_id, rois)

        # 1.5  Build the list of objects that the Slides UI can consume
        annotations = []

        # Polygons ---------------------------------------------------------------
        if output_type == "polygons":
            anno_polygons = convert_2_anno2_uuid(
                result_polygons,
                client,
                metadata='{ "comment": "dark polygons"}',
            )
            annotations.append({
                "type": "anno2",
                "name": "anno2 dark polygons",
                "value": anno_polygons,
                "color": "#00FF00",
            })

        # Points ---------------------------------------------------------------
        if output_type == "points":
            points = convert_polygons_2_centroids(result_polygons)
            anno_points = convert_2_anno2_uuid(
                points,
                client,
                metadata='{ "comment": "dark points"}',
            )
            annotations.append({
                "type": "anno2",
                "name": "anno2 dark points",
                "value": anno_points,
                "color": "#FFFF00",
            })

        # Heat‑map ---------------------------------------------------------------
        if output_type == "heatmap":
            points = convert_polygons_2_centroids(result_polygons)
            heatmap = convert_points_2_heatmap(points)
            anno_heatmap = convert_2_anno2_uuid(
                [heatmap],
                client,
                metadata='{ "comment": "heatmap of dark points"}',
            )
            annotations.append({
                "type": "anno2",
                "name": "anno2 heatmap",
                "value": anno_heatmap,
                "color": "Turbo",
            })

        # Description -------------------------------------------------------------
        elapsed = time.time() - request_json.get("start_time", time.time())
        annotations.append({
            "type": "text",
            "name": "Description of results",
            "value": f'These results took {elapsed:.2f} s to generate',
        })

        # 1.6  Persist the finished job ------------------------------------------
        async with job_lock:
            job_store[job_id] = {
                "status": "done",
                "result": annotations,
            }

    except Exception as exc:          # pragma: no cover
        # 1.7  Error handling ------------------------------------------------------
        traceback.print_exc()
        async with job_lock:
            job_store[job_id] = {
                "status": "error",
                "message": str(exc),
            }


# --------------------------------------------------------------
# 2.  handle_check - return status or full result
# --------------------------------------------------------------
async def handle_check(request):
    """
    HTTP GET /check?id=<job_id> → JSON object

    * If the job is still running - returns
        {"status":"running", "id":"<job_id>"}

    * If the job finished - returns
        {
            "status":"done",
            "result":[ ... ]          # the full array of annotations
        }

    * If an error happened - returns
        {"status":"error", "message":"…"}
    """
    # --------------------------------------------------------------
    # 2.1  Extract job id from query string or URL path
    # --------------------------------------------------------------
    job_id = request.rel_url.query.get("id") or request.match_info.get("id")
    if not job_id:
        # Missing job id - bad request
        return web.json_response(
            {"status": "error", "message": "Missing job id"},
            status=400,
        )

    # --------------------------------------------------------------
    # 2.2  Look the job up in the in‑memory store
    # --------------------------------------------------------------
    async with job_lock:
        job = job_store.get(job_id)

    if not job:
        # Unknown job id - not found
        return web.json_response(
            {"status": "error", "message": "Unknown job id"},
            status=404,
        )

    # --------------------------------------------------------------
    # 2.3  Build the response according to the job status
    # --------------------------------------------------------------
    status = job["status"]
    response = {"status": status}

    if status == "done":
        # Job finished - attach the full result array in the output property
        response["output"] = job.get("result", [])
    elif status == "error":
        # Job finished with an error - report it
        response["message"] = job.get("message", "Unknown error")
    else:
        # Job is still running - we can also expose the job id
        response["id"] = job_id

    return web.json_response(response)



def init_app():
    app = web.Application()
    app.router.add_post("/", handle_post)
    app.router.add_get("/check", handle_check)
    app.router.add_get("/check/{id}", handle_check)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SlideScore async webhook for multi‑slide analysis",
        description=DESC,
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="HOST to listen on"
    )
    parser.add_argument(
        "--port", type=int, default=8101, help="PORT to listen on"
    )

    args = parser.parse_args()

    app = init_app()
    logging.basicConfig(level=logging.INFO)
    web.run_app(app, host=args.host, port=args.port)
