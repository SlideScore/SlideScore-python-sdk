"""
This test checks the ability to create a study by uploading results to a newly created study
verifying they are saved and making sure invalid results raise an error.
"""

import sys
import os
from datetime import datetime
import tempfile

import slidescore
from common_lib import create_study


def create_answers(image_id: int, image_name: str, email: str):
    i = 1
    # Add answer to the freetext question and the option question
    freetext_answer = slidescore.SlideScoreResult({
        "id": -1,
        "imageID": image_id,
        "imageName": image_name,
        "user": email,
        "question": "Test question",
        "answer": f"test answer {i}"
    })

    option_answer = slidescore.SlideScoreResult({
        "id": -1,
        "imageID": image_id,
        "imageName": image_name,
        "user": email,
        "question": "Options question",
        "answer": f"Option{(i % 3) + 1}"
    })
    answers = [freetext_answer, option_answer]
    return answers


def test_upload_study_results():
    SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
    SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
    USER_EMAIL = os.getenv('SLIDESCORE_EMAIL') or "pytest@example.com"

    # Make sure we got a HOST and KEY
    assert SLIDESCORE_HOST
    assert SLIDESCORE_API_KEY
    assert USER_EMAIL
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)

    # Prepare a study and some sample answers
    datetime_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    study_name = f'test-study-upload-questions-{datetime_str}'

    question_str = 'Test question	FreeText\nOptions question	ClickFriendlyPollOnly	Option1;;Option2;;Option3'
    study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL, question_str)
    answers = create_answers(image_id, image_name, USER_EMAIL)

    # Upload the answers
    has_uploaded = client.upload_results(study_id, answers)
    assert has_uploaded

    # Now retrieve them from the API and check they are the same
    retrieved_results = client.get_results(study_id)

    # Sort both lists so they are guaranteed to be in the same order
    answers.sort(key=lambda r: r.toRow())
    retrieved_results.sort(key=lambda r: r.toRow())
    print(answers)
    print(retrieved_results)
    for answer_local, answer_remote in zip(answers, retrieved_results):
        assert answer_local.toRow() == answer_remote.toRow()

    # Now make sure we can upload the results again
    assert client.upload_results(study_id, answers)

    # Now check we cannot upload an invalid result
    try:
        invalid_answer = slidescore.SlideScoreResult({
            "imageID": image_id,
            "imageName": image_name,
            "user": email,
            "question": "Non-existant question",
            "answer": "answer"
        })
        client.upload_results(study_id, [invalid_answer])
        assert False # Succeeded?
    except:
        assert True # Failed succesfully!

if __name__ == "__main__":
    sys.exit('This file is meant to be ran by PyTest')