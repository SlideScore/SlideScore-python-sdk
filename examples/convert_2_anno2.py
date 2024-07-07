"""
This example uses a simple API call to convert an existing score stored in the SlideScore
database into the high performance and low disk usage binary format.

Usage: python examples/convert_2_anno2.py 4
Date: 3-3-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os

import slidescore

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/

def get_score_value_id(client: slidescore.APIClient):
    """Helper function to allow the user to select a study and image and AnnoShape answer."""
    available_studies = client.get_studies()
    if len(available_studies) == 0:
        exit("Please create a study with a slide before running this example.")
    
    # Get study
    print("\nStudy options:")
    for study in available_studies:
        print(f'{study["id"]}: {study["name"]}')

    study_id = int(input('Which study number to use: '))

    # Verify it has the correct annotation question that we want to answer
    study_questions = client.perform_request("Questions", {"studyId": study_id}, method="GET").json()
    anno_shape_questions = [q for q in study_questions if q["typeName"] == "AnnoShapes" or q["typeName"] == "AnnoPoints"]
        
    if len(anno_shape_questions) == 0:
        exit("This study does not have a question with the Shapes type, please add it")
    
    # Get image
    images = client.get_images(study_id)
    print("\nImage options:")
    for image in images:
        print(f'{image["id"]}: {image["name"]}')

    image_id = int(input('Which image number to use: '))

    # Get question
    for i, question in enumerate(anno_shape_questions):
        print(f'{i}: {question["name"]}')

    question_i = int(input('Which question number to use: '))
    question_name = anno_shape_questions[question_i]["name"]

    # Get scorevalue id
    scores = client.get_results(study_id, question_name, None, image_id)

    if len(scores) == 0:
        exit(f"The question '{question_name}' does not have a scores, please answer it first")

    for score in scores:
        print(f'{score.id}: {score.user}')

    score_value_id = int(input('Which score number to use: '))
    
    return score_value_id


if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    score_value_id = int(sys.argv[1]) if len(sys.argv) > 1 else get_score_value_id(client)
    metadata = '{ "description": "Hello World" }'
    resp = client.perform_request("ConvertScoreValueToAnno2", {"scoreValueId": score_value_id, "metadata": metadata}, method="POST")
    print("Conversion response: ", resp.text)

    print(f'Done')