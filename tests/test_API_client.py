"""
This test checks the ability to create a API client and the reachability of the host
"""

import sys
import os

import slidescore

# Either set the environment variables, or hardcode your settings below

def test_client_creation():
    SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/

    # Make sure we got a HOST and KEY
    assert SLIDESCORE_HOST
    assert SLIDESCORE_API_KEY
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    response = client.perform_request('Studies', None, method="GET")
    assert response.status_code == 200


if __name__ == "__main__":
    sys.exit('This file is meant to be ran by PyTest')