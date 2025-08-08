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

from aiohttp import web # pip install aiohttp
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
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Performed local image analysis")

        cur_img_dims = (img.shape[1], img.shape[0])
        roi_polygons = convert_contours_2_polygons(contours, cur_img_dims, roi)
        polygons += roi_polygons
        print("Converted image analysis results to SlideScore annotation")

    return polygons


# --------------------------------------------------------------------------- #
# Async server implementation
# --------------------------------------------------------------------------- #

# Global job store – maps job_id to status and optional result
job_store = {}
job_lock = asyncio.Lock()


async def handle_post(request):
    """Handle incoming POST - start background job and return job id."""
    try:
        post_body = await request.text()
        request_json = json.loads(post_body)
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


async def process_job(job_id, request_json):
    """Background worker that performs the heavy analysis."""
    try:
        host = request_json["host"]
        study_id = int(request_json["studyid"])
        image_id = int(request_json["imageid"])
        imagename = request_json["imagename"]
        case_id = int(request_json["caseid"])
        email = request_json["email"]
        analysis_id = int(request_json["analysisid"])
        analysis_name = request_json["analysisname"]
        case_name = request_json["casename"]
        answers = request_json["answers"]
        apitoken = request_json["apitoken"]

        client = slidescore.APIClient(host, apitoken)

        img_metadata = client.perform_request(
            "GetImageMetadata", {"imageid": image_id}, method="GET"
        ).json()["metadata"]

        img_width, img_height = img_metadata["level0Width"], img_metadata["level0Height"]

        # Run the heavy analysis synchronously – this is fine because it runs in a background task
        result_polygons = threshold_image(
            client,
            image_id,
            [{"corner": {"x": 0, "y": 0}, "size": {"x": img_width, "y": img_height}}]
        )

        # Convert results to anno2 UUIDs
        anno2_polygons = convert_2_anno2_uuid(
            result_polygons, client, metadata='{ "comment": "dark polygons"}'
        )

        # Store final result – only status is required by the client, but we keep the data for debugging
        async with job_lock:
            job_store[job_id] = {
                "status": "done",
                "result": {
                    "polygons": anno2_polygons,
                    "description": f"Results generated in {(time.time() - request_json.get('start_time', time.time())):.2f}s"
                }
            }

    except Exception as e:
        traceback.print_exc()
        async with job_lock:
            job_store[job_id] = {"status": "error", "message": str(e)}


async def handle_check(request):
    """Return the status of a job."""
    job_id = request.rel_url.query.get("id") or request.match_info.get("id")
    if not job_id:
        return web.json_response(
            {"status": "error", "message": "Missing job id"}, status=400
        )

    async with job_lock:
        job = job_store.get(job_id)

    if not job:
        return web.json_response(
            {"status": "error", "message": "Unknown job id"}, status=404
        )

    # Return status and optionally result if done
    response = {"status": job["status"], "id": job_id}
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["message"] = job.get("message")

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
        description=DESC
    )
    parser.add_argument("--host", type=str, default="localhost", help="HOST to listen on")
    parser.add_argument("--port", type=int, default=8101, help="PORT to listen on")

    args = parser.parse_args()

    app = init_app()
    logging.basicConfig(level=logging.INFO)
    web.run_app(app, host=args.host, port=args.port)
