DESC = """
TODO

Date: 24-5-2024
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import traceback
import time
import math
from io import BytesIO
import sys

import slidescore

import cv2 # $ pip install opencv-python
import numpy as np # $ pip install numpy

import torch
from fastsam import FastSAM, FastSAMPrompt
from PIL import Image

print("Loading model...")
model = FastSAM('./FastSAM.pt')

device_string = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(device_string)
print(f"USING TORCH DEVICE = {DEVICE}")

save_images = False

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


def get_level(client, image_id, width, height, desired_size = 1024):
    # Calculate the DeepZoom level 
    img_metadata = client.perform_request("GetImageMetadata", {"imageid": image_id}, method="GET").json()["metadata"]
    img_width, img_height = img_metadata["level0Width"], img_metadata["level0Height"]
    max_level = math.ceil(math.log2(max(img_width, img_height)))
    max_level_api = max_level + 2
    print(f"Detected image with {max_level_api} levels")

    requested_size = max(width, height)
    log_diff = math.log2(requested_size / desired_size)
    num_times_zoomed_out = round(log_diff)
    return max_level_api - num_times_zoomed_out

def parse_points(roi, cur_img_dims, positive_points, negative_points):
    # Convert lists of roi, positive points, negative points with absolute coords to list of
    # [relative_point_coords, point_labels]
    points = [] # [(626, 561), (228, 510), (546, 520), (287, 750)]
    point_labels = [] # [1, 1, 0, 1] # 1 positive, 0 negative
    
    x_factor = roi["size"]["x"] / cur_img_dims[0]
    y_factor = roi["size"]["y"] / cur_img_dims[1]
    x_offset = roi["corner"]["x"]
    y_offset = roi["corner"]["y"]

    all_points = list(zip(positive_points, [1] * len(positive_points))) + list(zip(negative_points, [0] * len(negative_points)))

    for point, label in all_points:
        orig_x, orig_y = int(point["x"]), int(point["y"])
        if orig_x < roi["corner"]["x"] or orig_x > roi["size"]["x"] + roi["corner"]["x"]:
            continue
        if orig_y < roi["corner"]["y"] or orig_y > roi["size"]["y"] + roi["corner"]["y"]:
            continue

        roi_x, roi_y = int((orig_x - x_offset) / x_factor), int((orig_y - y_offset) / y_factor)
        points.append((roi_x, roi_y))
        point_labels.append(label)

    return points, point_labels


def convert_contours_2_polygons(contours, cur_img_dims, roi):
    """Converts OpenCV2 contours to AnnoShape Polygons format of SlideScore
    Also needs the original img width and height to properly map the coordinates"""

    x_factor = roi["size"]["x"] / cur_img_dims[0]
    y_factor = roi["size"]["y"] / cur_img_dims[1]
    x_offset = roi["corner"]["x"]
    y_offset = roi["corner"]["y"]

    polygons = []
    for contour in contours:
        points = []
        for point in contour:
            # The contours are based on a scaled down version of the image
            # so translate these coordinates to coordinates of the original image
            orig_x, orig_y = int(point[0][0]), int(point[0][1])
            points.append({"x": x_offset + int(x_factor * orig_x), "y": y_offset + int(y_factor * orig_y)})
        polygon = {
            "type":"polygon",
            "points": points
        }
        polygons.append(polygon)
    return polygons

def segment_image(client, image_id: int, rois: list, positive_points: list, negative_points: list, conf=0.05, iou = 0.1):
    # Extract pixel information by making a "screenshot" of each region of interest
    polygons = []

    rois = [roi for roi in rois if roi["corner"]["x"] is not None and roi["corner"]["y"] is not None]
    if len(rois) == 0:
        raise Exception("Failed to add valid roi")
    
    for roi in rois:
        dzi_level = get_level(client, image_id, roi["size"]["x"], roi["size"]["y"], 1500)
        image_response = client.perform_request("GetScreenshot", {
            "imageid": image_id,
            "x": roi["corner"]["x"],
            "y": roi["corner"]["y"],
            "width": roi["size"]["x"],
            "height": roi["size"]["y"],
            "level": dzi_level,
            "showScalebar": "false"
        }, method="GET")
        jpeg_bytes = image_response.content
        print("Retrieved image from server, performing analysis using FastSAM")

        # Create a BytesIO object from the JPEG bytes
        image_stream = BytesIO(jpeg_bytes)

        # Open the image using Pillow
        input_image = Image.open(image_stream)
        input_image = input_image.convert("RGB")
        cur_img_dims = input_image.size

        everything_results = model(
            input_image,
            device=device_string,
            retina_masks=False,
            imgsz=1024,
            conf=conf, # object confidence threshold, default = 0.9
            iou=iou # iou threshold for filtering the annotations, default = 0.9
        )
        prompt_process = FastSAMPrompt(input_image, everything_results, device=device_string)

        points = None
        point_labels = None
        if len(positive_points) == 0 and len(negative_points) == 0:
            annotations = prompt_process.everything_prompt()
        else:
            points, point_labels = parse_points(roi, cur_img_dims, positive_points, negative_points)
            if len(points) == 0:
                annotations = prompt_process.everything_prompt()
            else:
                annotations = prompt_process.point_prompt(
                    points=points, pointlabel=point_labels
                )

        if save_images:
            # Save output
            prompt_process.plot(
                annotations=annotations,
                output_path=f'output_{int(time.time())}.jpg',
                bboxes = None,
                points = points,
                point_label = point_labels,
                withContours = True,
                better_quality = True,
            )
            print("Saved output image to ", f'output_{int(time.time())}.jpg')

        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        all_contours = []
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']
            annotation = mask.astype(np.uint8)
            # Mask has been resized to 1024x1024 by model, size it back to the original size of the image
            annotation = cv2.resize(annotation, input_image.size, interpolation=cv2.INTER_NEAREST)
            contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                all_contours.append(contour)
        print("Performed local image analysis")

        # Convert OpenCV2 contour to AnnoShape Polygons format of SlideScore
        roi_polygons = convert_contours_2_polygons(all_contours, cur_img_dims, roi)
        polygons += roi_polygons
        print("Converted image analysis results to SlideScore annotation")

    device = torch.cuda.get_device_name() if "cuda" in device_string else device_string
    # AnnoShape polygons
    return polygons, dzi_level, device

def get_rois(answers: list):
    roi_json = next((answer["value"] for answer in answers if answer["name"] == "ROI"), None)
    if roi_json is None:
        raise Exception("Failed to find the ROI answer")
    rois = json.loads(roi_json)
    if len(rois) == 0:
        raise Exception("No ROI given")
    return rois

def get_points(answers: list):
    positive_points_json = next((answer["value"] for answer in answers if answer["name"] == "Positive points"), None)
    if positive_points_json is None or positive_points_json == '':
        print("Failed to find the 'Positive points' answer", file=sys.stderr)
    positive_points = json.loads(positive_points_json) if positive_points_json else []

    negative_points_json = next((answer["value"] for answer in answers if answer["name"] == "Negative points"), None)
    if negative_points_json is None or negative_points_json == '':
        print("Failed to find the 'Negative points' answer", file=sys.stderr)
    negative_points = json.loads(negative_points_json) if negative_points_json else []

    return positive_points, negative_points

def request_handler(request):
    time_got_request = time.time()

    host = request["host"]
    image_id = int(request["imageid"])
    answers = request["answers"] # Answers to the questions field, needs to be validated to contain the expected vals
    apitoken = request["apitoken"] # Api token that is generated on the fly for this request
    rois = get_rois(answers) # Get Regions Of Interest
    positive_points, negative_points = get_points(answers)

    client = slidescore.APIClient(host, apitoken)

    result_polygons, dzi_level, device = segment_image(client, image_id, rois, positive_points, negative_points)
    # [{type: "polygon", points: [{x: 1, y, 1}, ...]}]
    
    request['apitoken'] = "HIDDEN"
    print('Succesfully contoured image', request)

    # Give up token, cannot be used after this request
    client.perform_request("GiveUpToken", {}, "POST")

    # Return an JSON array with a single result, A list of polygons surrounding the dark parts of the ROI.
    return([{
        "type": "polygons", 
        "name": "SAM Results", 
        "value": result_polygons,
        "color": "#0000FF"
    },{
        "type": "polygons", 
        "name": "Input ROI's", 
        "value": rois_to_polygons(rois),
        "color": "#FF0000"
    },
    {
        "type": "points", 
        "name": "Positive points", 
        "value": positive_points,
        "color": "#00FF00"
    },
    {
        "type": "points", 
        "name": "Negative points", 
        "value": negative_points,
        "color": "#FF0000"
    },
    {
        "type": "text",
        "name": "Metadata",
        "value": f'These results took {(time.time() - time_got_request):.2f} s to generate using level {dzi_level} and device {device}'
    }])


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
            results = request_handler(request)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            # Return an JSON array with a single result, A list of polygons surrounding the dark parts of the ROI.

            self.wfile.write(bytes(json.dumps(results), "utf-8"))
        except Exception as e:
            print("Caught exception:", e)
            print(traceback.format_exc())

            print(post_body)
            self.send_response(500)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Unknown error: " + str(e), "utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='SlideScore openslide OOF detector API',
                    description=DESC)
    parser.add_argument('--host', type=str, default='localhost', help='HOST to listen on')
    parser.add_argument('--port', type=int, default=8000, help='PORT to listen on')
    parser.add_argument('--save-images', action='store_true', help='Store image results for debugging')

    args = parser.parse_args()
    if args.save_images:
        save_images = True

    webServer = HTTPServer((args.host, args.port), ExampleAPIServer)
    print(f"Server started http://{args.host}:{args.port}, configure your slidescore instance with a default analysis endpoint pointing to this host.")
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

