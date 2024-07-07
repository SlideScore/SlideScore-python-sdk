"""
TODO add description and perhaps study selection
"""
import sys
import os

import slidescore

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/

if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    study_id = int(input("What is the study id: "))
    images = client.get_images(study_id)

    print("Got images:")
    for image in images:
        size_in_mb = round(image["fileSize"] / (1024 * 1024))
        print(f"id: {image['id']} - {image['name']} - {size_in_mb} MB")
    while True:
        try:
            image_id = int(input("Image id to download (Ctrl+C to exit): "))
            client.download_slide(study_id, image_id, './')
        except KeyboardInterrupt:
            print("\nExiting")
            exit()