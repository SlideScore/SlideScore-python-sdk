"""
This example shows how to upload a locally created high performance annotation (anno2) to a specific 
image as a question answer.

Usage: python examples/upload_anno2.py points.zip
Date: 7-4-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os

import slidescore
import tempfile

def get_study_and_image(client: slidescore.APIClient):
    """Helper function to allow the user to set a study, image and question for anno2 upload."""
    available_studies = client.get_studies()
    if len(available_studies) == 0:
        exit("Please create a study with a slide before running this example.")
    
    print("\nStudy options:")
    for study in available_studies:
        print(f'{study["id"]}: {study["name"]}')

    study_id = int(input('Which study number to use: '))

    # Verify it has the correct annotation question that we want to answer
    study_questions = client.perform_request("Questions", {"studyId": study_id}, method="GET").json()

    question_options = [q for q in study_questions if q["typeName"] == "AnnoShapes" or q["typeName"] == "AnnoPoints"]
    
    if len(question_options) == 0:
        exit("This study does not have a question with the type AnnoShapes or AnnoPoints, please add it")
    print('[✓] Verified this study has a question option!')

    print("\nQuestion options:")
    for i, q in enumerate(question_options):
        print(f'{i}: {q["name"]}')
    
    selected_question_i = int(input('Which question number to use: '))
    question_name = question_options[selected_question_i]['name']


    images = client.get_images(study_id)
    print("\nImage options:")
    for image in images:
        print(f'{image["id"]}: {image["name"]}')

    image_id = int(input('Which image number to use: '))
    return study_id, image_id, question_name


# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('What is your Slidescore email: ') # test@example.com

if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    if len(sys.argv) < 2 or not sys.argv[1].endswith('.zip'):
        exit("Please supply an anno2 .zip file as the first argument")
    local_anno2_path = sys.argv[1]
    
    metadata = { # Set any JSON as metadata
        "origFilePath": local_anno2_path,
        "comment": "Created using example/upload_anno2.py"
    }

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    study_id, image_id, question_name = get_study_and_image(client)
    print("got studyid, imageid", study_id, image_id)
        
    # Convert to anno2
    print("Starting upload anno2 @", local_anno2_path, "with size:", int(os.path.getsize(local_anno2_path) / 1024), 'kiB')

    # Create DB entry serverside
    resp = client.perform_request("CreateAnno2", { 
        "studyid": study_id,
        "imageId": image_id,
        "question": question_name,
        "email": SLIDESCORE_EMAIL
    }, method="POST").json()

    print("Created an anno2 DB entry, uploading...")
    # Actually upload the annotation
    client.upload_using_token(local_anno2_path, resp["uploadToken"])
    
    print(f'Uploaded with uuid: {resp["annoUUID"]}')
    print(f'Done, view results at: {SLIDESCORE_HOST}/Image/Details?imageId={image_id}&studyId={study_id}')
