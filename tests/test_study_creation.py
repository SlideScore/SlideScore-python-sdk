"""
This test checks the ability to create a study by uploading config files, 
adding an example slide and verifing the study is created.
"""

import sys
import os
from datetime import datetime

import slidescore

from common_lib import create_study

def test_create_study():
    SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    USER_EMAIL = os.getenv('SLIDESCORE_EMAIL') or "pytest@example.com"

    # Make sure we got a HOST and KEY
    assert SLIDESCORE_HOST
    assert SLIDESCORE_API_KEY
    assert USER_EMAIL
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    
    datetime_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    study_name = f'test-study-{datetime_str}'
    study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL)

    # Now list all studies and check we got one with the currently imported id
    available_studies = client.get_studies()
    just_created_study = next((s for s in available_studies if s["id"] == study_id), None)
    assert just_created_study
    assert just_created_study['name'] == study_name
    
    images = client.get_images(study_id)
    assert len(images) == 1
    assert images[0]['name'] == 'test_slide'

if __name__ == "__main__":
    sys.exit('This file is meant to be ran by PyTest')