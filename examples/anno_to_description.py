"""
This example converts existing annotation answers stored to slide description for a training study in the SlideScore
It demonstrates the uses of API calls for getting results, study configuration and setting slide description. It also 
shows how to convert point annotations to shapes (ellipses) and how to construct a link that shows an annotation when clicked

Author: Bart Grosman & Jan Hudecek (SlideScore B.V.)
"""

import sys
import os
import json

from urllib.parse import quote
from html import escape


import slidescore

#provide these parameters
studyid=42
user='jan@slidescore.com'


# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/

def update_desc(image_id, desc):
     resp = client.perform_request("SetSlideDescription", {"imageId": image_id, 'studyId':studyid, "description": desc}, method="POST")
     resp.json()

def convert_points(points):
    return [
        {"center": point, "size": {"x": 100, "y": 100}, "type": "ellipse"} 
        for point in points
    ]

if __name__ == "__main__":
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    question_config = client.get_config(studyid)["scores"]
    question_colors = { q["name"]:q["valuesAllowed"].split(';')[0] for q in question_config}

    results = client.get_results(studyid, email=user)
    
    descs = {}
    for r in results:
        if r.image_id not in descs:
            descs[r.image_id] = ""
        if not r.answer.startswith('[{'): 
            continue
        annos=json.loads(r.answer)
        if 'x' in annos[0]:
            annos = convert_points(annos)
        shapes = '[{"color":"'+question_colors[r.question]+'","shapes":'+ json.dumps(json.dumps(annos))+'}]'
        link = SLIDESCORE_HOST+'/Image/Details?imageId='+str(r.image_id)+'&studyId='+str(studyid)+'&annos='+quote(shapes);
        html = '<a style="display: inline-block;" class="jsSquireAnnoLink jsSquireLink" data-target-image="'+str(r.image_id)+'" data-target-height="1" data-target-width="1" data-target-y="0" data-target-x="0" rel="noopener" href="'+link+'">'+escape(r.question)+'</a><br>'
        descs[r.image_id] = descs[r.image_id] + html
        
    for image_id in descs:
        update_desc(image_id, descs[image_id])
    

    print(f'Done')



