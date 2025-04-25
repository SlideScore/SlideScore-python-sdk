DESC = """
TODO

Date: 24-5-2024
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import tempfile
import traceback
import time

import slidescore

import cv2 # $ pip install opencv-python
import numpy as np # $ pip install numpy

def create_tmp_file(content: str, suffix='.tmp'):
    """Creates a temporary file, used for intermediate files"""
    
    fd, name = tempfile.mkstemp(suffix)
    if content:
        with open(fd, 'w') as fh:
            fh.write(content)
    return name

def convert_2_anno2_uuid(items, client, metadata=''):
    # Convert to anno2 zip, upload, and return uploaded anno2 uuid
    local_anno2_path = create_tmp_file('', '.zip')
    client.convert_to_anno2(items, metadata, local_anno2_path)
    response = client.perform_request("CreateOrphanAnno2", {}, method="POST").json()
    assert response["success"] is True

    client.upload_using_token(local_anno2_path, response["uploadToken"])
    return response["annoUUID"]

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

def convert_points_2_heatmap(points, size_per_pixel = 64):
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

def threshold_image(client, image_id: int, rois: list):
    # Extract pixel information by making a "screenshot" of each region of interest
    polygons = []
    for roi in rois:
        if roi["corner"]["x"] is None or roi["corner"]["y"] is None:
            continue # Basic validation
    
        image_response = client.perform_request("GetScreenshot", {
            "imageid": image_id,
            "x": roi["corner"]["x"],
            "y": roi["corner"]["y"],
            "width": roi["size"]["x"],
            "height": roi["size"]["y"],
            "level": 15,
            "showScalebar": "false"
        }, method="GET")
        jpeg_bytes = image_response.content
        print("Retrieved image from server, performing analysis using OpenCV")

        # Parse the returned JPEG using OpenCV, and extract the contours from it.
        treshold = 220
        jpeg_as_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(jpeg_as_np, flags=1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, treshold, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Performed local image analysis")

        # Convert OpenCV2 contour to AnnoShape Polygons format of SlideScore
        cur_img_dims = (img.shape[1], img.shape[0])
        roi_polygons = convert_contours_2_polygons(contours, cur_img_dims, roi)
        polygons += roi_polygons
        print("Converted image analysis results to SlideScore annotation")

    # AnnoShape polygons
    return polygons

def get_rois(answers: list):
    roi_json = next((answer["value"] for answer in answers if answer["name"] == "ROI"), None)
    if roi_json is None:
        raise Exception("Failed to find the ROI answer")
    rois = json.loads(roi_json)
    if len(rois) == 0:
        raise Exception("No ROI given")
    return rois

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
            time_got_request = time.time()
            """
            default http post payload:
                "host": "${document.location.origin}",
                "studyid": %STUDY_ID%,
                "imageid": %IMAGE_ID%,
                "imagename": "%IMAGE_NAME%",
                "caseid": %CASE_ID%,
                "casename": "%CASE_NAME%",
                "email": "%USER_EMAIL%",
                "analysisid": %ANALYSIS_ID%,
                "analysisname": "%ANALYSIS_NAME%",
                "answers": %ANSWERS%,
                "apitoken": "%API_TOKEN%"
            """
            host = request["host"]
            study_id = int(request["studyid"])
            image_id = int(request["imageid"])
            imagename = request["imagename"]
            case_id = int(request["imageid"])
            email = request["email"]
            analysis_id = int(request["analysisid"])
            analysis_name = request["analysisname"]
            case_name = request["casename"]
            answers = request["answers"] # Answers to the questions field, needs to be validated to contain the expected vals
            apitoken = request["apitoken"] # Api token that is generated on the fly for this request
            rois = get_rois(answers) # Get Regions Of Interest
            
            client = slidescore.APIClient(host, apitoken)

            result_polygons = threshold_image(client, image_id, rois)
            # [{type: "polygon", points: [{x: 1, y, 1}, ...]}]
            
            request['apitoken'] = "HIDDEN"
            print('Succesfully contoured image', request)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            # Return an JSON array with a single result, A list of polygons surrounding the dark parts of the ROI.
            points = convert_polygons_2_centroids(result_polygons)
            # Convert centroids to a heatmap
            heatmap = convert_points_2_heatmap(points)

            self.wfile.write(bytes(json.dumps([{
                "type": "polygons", 
                "name": "Dark parts", 
                "value": result_polygons,
                "color": "#0000FF"
            }, {
                "type": "points",
                "name": "Dark parts centroids",
                "value": points,
                "color": "#00FFFF"
            }, {
                "type": "anno2",
                "name": "anno2 dark polygons",
                "value": convert_2_anno2_uuid(result_polygons, client, metadata='{ "comment": "dark polygons"}'),
                "color": "#00FF00"
            }, {
                "type": "anno2",
                "name": "anno2 dark points",
                "value": convert_2_anno2_uuid(points, client, metadata='{ "comment": "dark points"}'),
                "color": "#FFFF00"
            },
            {
                "type": "anno2",
                "name": "anno2 heatmap",
                "value": convert_2_anno2_uuid([heatmap], client, metadata='{ "comment": "heatmap of dark points"}'),
                "color": "Turbo"
            },
            {
                "type": "text",
                "name": "Description of results",
                "value": f'These results took {(time.time() - time_got_request):.2f} s to generate'
            }
            ]), "utf-8"))

            # Give up token, cannot be used after this request
            client.perform_request("GiveUpToken", {}, "POST")           

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

    args = parser.parse_args()

    webServer = HTTPServer((args.host, args.port), ExampleAPIServer)
    print(f"Server started http://{args.host}:{args.port}, configure your slidescore instance with a default analysis endpoint pointing to this host.")
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

