"""
This test checks the ability to upload asap annotation file and generate questions
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
        "imageID": image_id,
        "imageName": image_name,
        "user": email,
        "question": "Test question",
        "answer": f"test answer {i}"
    })

    option_answer = slidescore.SlideScoreResult({
        "imageID": image_id,
        "imageName": image_name,
        "user": email,
        "question": "Options question",
        "answer": f"Option{(i % 3) + 1}"
    })
    answers = [freetext_answer, option_answer]
    return answers


def test_upload_asap():
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
    study_name = f'test-study-questions-{datetime_str}'

    question_str = 'Test anno	AnnoShapes	#e6194b'
    study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL, question_str)
    
    asap = """<ASAP_Annotations>	<Annotations>	<Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="red" Color="#000000"><Coordinates><Coordinate Order="0" X="40444" Y="37114" /><Coordinate Order="1" X="40444" Y="37104" /><Coordinate Order="2" X="40454" Y="37104" /><Coordinate Order="3" X="40454" Y="37095" /><Coordinate Order="4" X="40543" Y="37095" /><Coordinate Order="5" X="40543" Y="37085" /><Coordinate Order="6" X="40563" Y="37085" /><Coordinate Order="7" X="40563" Y="37075" /><Coordinate Order="8" X="40573" Y="37075" /><Coordinate Order="9" X="40573" Y="37065" /><Coordinate Order="10" X="40583" Y="37065" /><Coordinate Order="11" X="40583" Y="37055" /><Coordinate Order="12" X="40602" Y="37055" /><Coordinate Order="13" X="40602" Y="37045" /><Coordinate Order="14" X="40632" Y="37045" /><Coordinate Order="15" X="40632" Y="37035" /></Coordinates></Annotation>
    <Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="red" Color="#e6194b"><Coordinates><Coordinate Order="0" X="40444" Y="37114" /><Coordinate Order="1" X="40444" Y="37104" /><Coordinate Order="2" X="40454" Y="37104" /><Coordinate Order="3" X="40454" Y="37095" /><Coordinate Order="4" X="40543" Y="37095" /><Coordinate Order="5" X="40543" Y="37085" /><Coordinate Order="6" X="40563" Y="37085" /><Coordinate Order="7" X="40563" Y="37075" /><Coordinate Order="8" X="40573" Y="37075" /><Coordinate Order="9" X="40573" Y="37065" /><Coordinate Order="10" X="40583" Y="37065" /><Coordinate Order="11" X="40583" Y="37055" /><Coordinate Order="12" X="40602" Y="37055" /><Coordinate Order="13" X="40602" Y="37045" /><Coordinate Order="14" X="40632" Y="37045" /><Coordinate Order="15" X="40632" Y="37035" /></Coordinates></Annotation>
    </Annotations></ASAP_Annotations>"""
    
    # Upload the answers
    result = client.perform_request("UploadASAPAnnotations", {
        
                 "imageId": image_id,
                 "user": USER_EMAIL,
                 "asapAnnotation": asap,
                 "questionsMap": "#e6194b;Test anno"
                 }, method="POST")

    assert result
    assert result.text[0] != '"'
    rjson=result.json()
    assert rjson["success"]


    # Now retrieve them from the API
    retrieved_results = client.get_results(study_id)



if __name__ == "__main__":
    sys.exit('This file is meant to be ran by PyTest')