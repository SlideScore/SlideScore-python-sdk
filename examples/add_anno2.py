"""
This example shows how to locally create a high performance annotation, and upload it to a specific 
image as a question answer.

Usage: python examples/add_anno2.py points.tsv
Date: 5-4-2023
Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os

import slidescore
import tempfile

def get_study_and_image(client: slidescore.APIClient):
    """Helper function to allow the user to set a study and image for analysis.
    Also checks if the study has the expected question of type AnnoShape"""
    available_studies = client.get_studies()
    if len(available_studies) == 0:
        exit("Please create a study with a slide before running this example.")
    
    print("\nStudy options:")
    for study in available_studies:
        print(f'{study["id"]}: {study["name"]}')

    study_id = int(input('Which study number to use: '))

    # Verify it has the correct annotation question that we want to answer
    study_questions = client.perform_request("Questions", {"studyId": study_id}, method="GET").json()

    has_correct_question = False
    for question in study_questions:
        if question["name"] == "Annotate shape" and question["typeName"] == "AnnoShapes":
            has_correct_question = True
    
    if not has_correct_question:
        exit("This study does not have a question with the name: 'Annotate shape' of the Shapes type, please add it or change the code of the example")
    print('[✓] Verified this study has the correct question!')
    
    images = client.get_images(study_id)
    print("\nImage options:")
    for image in images:
        print(f'{image["id"]}: {image["name"]}')

    image_id = int(input('Which image number to use: '))
    return study_id, image_id


# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('What is your Slidescore email: ') # test@example.com

if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    if len(sys.argv) < 2 or not sys.argv[1].endswith('.tsv'):
        exit("Please supply a .tsv file as the first argument, 2 coords per line")
    tsv_file = sys.argv[1]
    metadata = { # Set any JSON as metadata
        "origFilePath": tsv_file,
        "comment": "Created using example/add_anno2.py"
    }

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    study_id, image_id = get_study_and_image(client)
    print("got studyid, imageid", study_id, image_id)

    with tempfile.TemporaryDirectory() as tmp_dirname:
        print('Created temporary directory for Anno2 storage', tmp_dirname)
        
        # Read the TSV data into memory
        anno_data = slidescore.lib.utils.read_tsv(tsv_file, "circles") #or mask, polygon, or brush

        # Convert to anno2
        local_anno2_path = os.path.join(tmp_dirname, 'anno.zip')
        client.convert_to_anno2(anno_data, metadata, local_anno2_path)
        print("Created anno2 @", local_anno2_path, "with size:", int(os.path.getsize(local_anno2_path) / 1024), 'kiB')

        # Create DB entry serverside
        anno2 = client.create_anno2(study_id=study_id, image_id=image_id, score_id=None, email=SLIDESCORE_EMAIL, case_id=None, tma_core_id=None, question="Annotate shape")
        newp = os.path.join(tmp_dirname, anno2['annoUUID'])
        os.rename(local_anno2_path, newp)
        print(f'Uploading {str(newp)} using {anno2["uploadToken"]}')
        # Actually upload the annotation
        client.upload_using_token(newp, anno2["uploadToken"])
        
        print(f'Uploaded with uuid: {anno2["annoUUID"]}')
        print(f'Done, view results at: {SLIDESCORE_HOST}/Image/Details?imageId={image_id}&studyId={study_id}')
