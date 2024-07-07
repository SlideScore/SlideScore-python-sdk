"""
This test checks the ability to upload a slide, and then download it again.
"""

import sys
import os
import tempfile
from datetime import datetime

import slidescore

from common_lib import create_study

def test_high_perf_img_server():
    SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    USER_EMAIL = os.getenv('SLIDESCORE_EMAIL') or "pytest@example.com"

    # Make sure we got a HOST and KEY
    assert SLIDESCORE_HOST
    assert SLIDESCORE_API_KEY
    assert USER_EMAIL
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)

    # Create study with slide
    datetime_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    study_name = f'test-study-download-{datetime_str}'
    study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL)
    
    # Download the slide from the server
    with tempfile.TemporaryDirectory() as tmp_dirname:
        client.download_slide(study_id, image_id, tmp_dirname)
        # Should be downloaded, check the dir
        dir_contents = os.listdir(tmp_dirname)
        assert len(dir_contents) == 1
        downloaded_fn = dir_contents[0]
        assert downloaded_fn.startswith(image_name)
    

if __name__ == "__main__":
    sys.exit('This file is meant to be ran by PyTest')