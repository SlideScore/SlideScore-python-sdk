"""
This example show how to upload files in two stages to a remote folder
called "test". First it requests an upload token, and then using this 
token it performs the upload and notifies the server is has finished.
Uses the first argument to the script as filepath to upload

Usage: python examples/get_image_server.py slide.svs
Date: 3-3-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
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
    file_path_to_upload = sys.argv[1]

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    upload_token = client.request_upload('./test/', os.path.basename(file_path_to_upload), None)
    print('Got TUS token:', upload_token, 'uploading', file_path_to_upload)
    client.upload_using_token(file_path_to_upload, upload_token)
    print('Done')