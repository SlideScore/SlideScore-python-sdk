"""
This test checks if the swagger endpoint loads, and if there is valid JSON served
"""
import os
import requests

def test_swagger_page():
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    swagger_page_url = f"{SLIDESCORE_HOST}/swagger"
    resp = requests.get(swagger_page_url, allow_redirects=True)
    assert resp.status_code == 200


def test_swagger_json():
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    swagger_page_url = f"{SLIDESCORE_HOST}/swagger/v1/swagger.json"
    resp = requests.get(swagger_page_url, allow_redirects=True)
    assert resp.status_code == 200
    resp_data = resp.json()
    assert 'openapi' in resp_data
    assert 'info' in resp_data
    assert 'paths' in resp_data


if __name__ == "__main__":
    sys.exit('This file is meant to be ran by PyTest')