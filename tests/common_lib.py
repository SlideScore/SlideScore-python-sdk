"""
This file does not contains tests, but simply helper functions that are used by the tests
"""
import slidescore
import tempfile
import os

def create_tmp_file(content: str):
    """Creates a temporary file, used for intermediate files"""
    tmp = tempfile.NamedTemporaryFile('w', delete=False)
    tmp.write(content)
    return tmp.name

def create_study(client: slidescore.APIClient, study_name: str, email: str, question_str = None):
    # First create a file with the emails that have access
    email_file_content = f'{email};canscore,canedit,cangetresults\n'
    email_file_path = create_tmp_file(email_file_content)
    client.upload_file(email_file_path, '', f'study.{study_name}.emails')
    
    if question_str:
        question_file_path = create_tmp_file(question_str)
        client.upload_file(question_file_path, '', f'study.{study_name}.scores')

    # Upload example slide
    cur_file_dir = os.path.abspath(os.path.dirname(__file__))
    example_slide_path = os.path.join(cur_file_dir, 'test_slide.png')
    client.upload_file(example_slide_path, study_name)
    
    # Import the study
    response = client.reimport(study_name)
    assert 'id' in response
    assert 'log' in response
    study_id = response['id']
    images = client.get_images(study_id)
    image_id = images[0]['id']
    image_name = images[0]['name']
    return study_id, image_id, image_name
