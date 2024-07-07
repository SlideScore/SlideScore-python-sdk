"""
This example showcases a local image job in the most straightforward manner.
It creates a sample study with a single image ('slide.png'). It then creates
a "screenshot" of this slide and performs local analysis on this screenshot file.
Using OpenCV 2 it extracts the dark parts of the image, it then converts these
OpenCV contours into SlideScore compatible annotations, and uploads them to the
server. Finally it prints out an URL that can be used to inspect the results.

Date: 3-3-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os
import tempfile
import datetime
import json

import slidescore

import cv2 # $ pip install opencv-python
import numpy as np # $ pip install numpy

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('What is your Slidescore email: ') # admin@example.com

def create_tmp_file(content: str):
    """Creates a temporary file, used for intermediate files"""
    tmp = tempfile.NamedTemporaryFile('w', delete=False)
    tmp.write(content)
    return tmp.name


def create_study(client: slidescore.APIClient, name: str, slide_paths, emails = [], questions_str = None):
    """Creates a study in slidescore using configuration files and local slide files"""
    # First create a file with the emails that have access
    email_file_content = '\n'
    for email in emails:
        # Give every email full access for this test study
        email_file_content += f'{email};canscore,canedit,cangetresults\n'
    email_file_path = create_tmp_file(email_file_content)
    client.upload_file(email_file_path, '', f'study.{name}.emails')
    
    # Then set questions if needed
    if questions_str is not None:
        questions_file_path = create_tmp_file(questions_str)
        client.upload_file(questions_file_path, '', f'study.{name}.scores')

    # Upload slides
    for slide_path in slide_paths:
        client.upload_file(slide_path, name)
    
    # Import the study
    response = client.reimport(name)
    study_id = response['id']
    log = response['log']
    return study_id, log
    
def convert_contours_2_polygons(contours, cur_img_dims, orig_img_dims):
    """Converts OpenCV2 contours to AnnoShape Polygons format of SlideScore
    Also needs the original img width and height to properly map the coordinates"""

    x_factor = orig_img_dims[0] / cur_img_dims[0]
    y_factor = orig_img_dims[1] / cur_img_dims[1]

    polygons = []
    for contour in contours:
        points = []
        for point in contour:
            # The contours are based on a scaled down version of the image
            # so translate these coordinates to coordinates of the original image
            orig_x, orig_y = int(point[0][0]), int(point[0][1])
            points.append({"x": int(x_factor * orig_x), "y": int(y_factor * orig_y)})
        polygon = {
            "type":"polygon",
            "points": points
        }
        polygons.append(polygon)
    return polygons

if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST or not SLIDESCORE_EMAIL:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST or SLIDESCORE_EMAIL not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    # Create a new example study with a single image and question
    now_date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_name = 'python-sdk-example_' + now_date_str
    slide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slide.png')

    anno_shape_questions_str = 'Annotate shape	AnnoShapes	#FF0000' # Create using the questions editor
    study_id, log = create_study(client, study_name, [slide_path], [SLIDESCORE_EMAIL], anno_shape_questions_str)
    print("Created study with id:", study_id)

    # Get image id
    images = client.get_images(study_id)
    image_id = images[0]["id"]

    # Extract pixel information by making a "screenshot" of the entire slide
    image_response = client.perform_request("GetScreenshot", {"imageid": image_id, "level": 11}, method="GET")
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
    img_metadata = client.perform_request("GetImageMetadata", {"imageid": image_id}, method="GET").json()["metadata"]
    orig_img_dims = (img_metadata["level0Width"], img_metadata["level0Height"])
    cur_img_dims = (img.shape[1], img.shape[0])
    polygons = convert_contours_2_polygons(contours, cur_img_dims, orig_img_dims)
    print("Converted image analysis results to SlideScore annotation")

    # Upload AnnoShape as answer to the question
    answer = slidescore.SlideScoreResult({
        "id": -1,
        "imageID": image_id,
        "imageName": "slide",
        "user": SLIDESCORE_EMAIL,
        "question": "Annotate shape",
        "answer": json.dumps(polygons)
    })
    client.upload_results(study_id, [answer])
    print("Uploaded image annotation")

    # Log URL you can use to easily view the result
    print(f'Done, view results at: {SLIDESCORE_HOST}/Image/Details?imageId={image_id}&studyId={study_id}' )