"""
GetScreenshot example

Usage: python examples/get_screenshot.py
Date: 7-6-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os

import slidescore

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('What is your Slidescore email: ') # test@example.com

def get_image_id(client: slidescore.APIClient):
    """Helper function to allow the user to select a study and image and AnnoShape answer."""
    available_studies = client.get_studies()
    if len(available_studies) == 0:
        exit("Please create a study with a slide before running this example.")
    
    # Get study
    print("\nStudy options:")
    for study in available_studies:
        print(f'{study["id"]}: {study["name"]}')

    study_id = int(input('Which study number to use: '))
    
    # Get image
    images = client.get_images(study_id)
    print("\nImage options:")
    for image in images:
        print(f'{image["id"]}: {image["name"]}')

    image_id = int(input('Which image number to use: '))
    
    return image_id


if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    image_id = int(sys.argv[1]) if len(sys.argv) > 1 else get_image_id(client)

    image_response = client.perform_request("GetScreenshot", {
        "imageid": image_id,
        "level": 14,
        "withAnnotationForUser": SLIDESCORE_EMAIL
    }, method="GET")
    jpeg_bytes = image_response.content
    with open('screenshot.jpg', 'wb') as fh:
        fh.write(jpeg_bytes)
    print("Done, retrieved image from server, stored as screenshot.jpg")
