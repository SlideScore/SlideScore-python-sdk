"""
This test checks if a local annotation can be converted into an anno2, and if it can be uploaded and screenshotted server-side
"""
import os
import tempfile
import shutil
from datetime import datetime
import json

from common_lib import create_study

import slidescore
import slidescore.bin_data


def gen_mask_tsv(size, x_offset=0, y_offset=0):
    tsv = ''
    middle = round(size / 2)
    square_size = round(size / 4)
    for x in range(size):
        for y in range(size):
            if x < (middle - square_size) or x > (middle + square_size) or y < (middle - square_size) or y > (middle + square_size):
                tsv += f'{x + x_offset}\t{y + y_offset}\n'
    return tsv

ANNO1_options = [
    {
        "filename": "points_anno1.json",
        "content": '[{"x": 10, "y": 10}, {"x": 15, "y": 15}, {"x": 20, "y": 20}]'
    },
    {
        "filename": "polygon.geojson", 
        "content": '{ "type": "FeatureCollection", "features": [ { "type": "Feature", "geometry": { "type": "Polygon", "coordinates": [ [ [132, 110], [298.97, 107.93], [300, 253], [125, 259] ] ] }, "properties": { "object_type": "annotation", "isLocked": false } } ]}'
    },
    {
        "filename": "polygon_anno1.json",
        "content": '[{"type":"polygon","modifiedOn":"2023-01-01T12:00:00.000Z","points":[{"x":132,"y":110},{"x":298,"y":107},{"x":300,"y":253},{"x":125,"y":259}],"area":"370 um2"}]'
    },
    {
        "filename": "polygon_anno1.tsv",
        "content": '132 110 298 107 300 253 125 259' 
    },
    {
        "filename": "heatmap.tsv",
        "content": """Heatmap 250 250 16 # x_offset y_offset size_per_pixel
0 0 128
1 1 255
2 1 200
3 1 150
4 1 100
5 1 50
6 1 0
7 1 5
"""
    },
    {
        "filename": "mask.tsv",
        "content": gen_mask_tsv(256, x_offset=400, y_offset=400),
        "extra_args": ['--points-type', 'mask'] 
    },
    {
        "filename": "points_png.json",
        "content": json.dumps([{"x": 50, "y": 50}] * 1000 * 1000),
    }
]
questions_str = "points	AnnoPoints	#e6194b\npolygons	AnnoShapes	#3cb44b\nheatmap	AnnoShapes	#3cb44b\nmask	AnnoPoints	#e6194b\npointsPNG	AnnoPoints	#00FF4b"
question_names = ['points', 'polygons', 'polygons', 'polygons', 'heatmap', 'mask', 'pointsPNG'] # See above


def create_anno2s():
    tmp_dirname = tempfile.mkdtemp('_slidescore')
    output_fns = []
    for option in ANNO1_options:
        # First write the Anno1 strings to disk
        option_file_path = os.path.join(tmp_dirname, option['filename'])
        option_anno2_path = os.path.splitext(option_file_path)[0] + '.zip'
        with open(option_file_path, 'w') as fh:
            fh.write(option['content'])
        
        # Then try to convert them
        args = ['-i', option_file_path, '-o', option_anno2_path]
        if 'extra_args' in option:
            args.extend(option['extra_args'])
        slidescore.bin_data.main(args)
        assert os.path.isfile(option_anno2_path)
        
        # Save the anno2 paths for later uploading
        output_fns.append(option_anno2_path)
    
    return output_fns

def test_anno2():
    # First test if we can create anno2s locally at all
    anno2_fns = create_anno2s()
    print("Created anno2 fns", anno2_fns)
    try:
        # Now do some setup to create a study with questions that we will upload the anno2s to
        SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') # eyb..
        SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') # https://slidescore.com/
        USER_EMAIL = os.getenv('SLIDESCORE_EMAIL') or "pytest@example.com"

        assert SLIDESCORE_HOST
        assert SLIDESCORE_API_KEY
        assert USER_EMAIL
        SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

        client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
        
        datetime_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        study_name = f'test-anno2-{datetime_str}'
        study_id, image_id, image_name = create_study(client, study_name, USER_EMAIL, questions_str)

        # Upload the first 2 anno2's
        for anno2_fn, question_name in zip(anno2_fns, question_names):
            # Notify the server we will upload an anno2
            resp = client.perform_request("CreateAnno2", { 
                "studyid": study_id,
                "imageId": image_id,
                "question": question_name,
                "email": USER_EMAIL
            }, method="POST").json()
            assert resp["uploadToken"]
            # Actually upload the annotation
            client.upload_using_token(anno2_fn, resp["uploadToken"])
        
        # Now check if we can create a screenshot
        image_response = client.perform_request("GetScreenshot", {
            "imageid": image_id,
            "level": 14,
            "withAnnotationForUser": USER_EMAIL
        }, method="GET")
        jpeg_bytes = image_response.content
        assert len(jpeg_bytes) > 1024

        with open('screenshot.jpg', 'wb') as fh:
            fh.write(jpeg_bytes)        
    finally:
        # Always remove the tmp dir
        tmp_dir = os.path.dirname(anno2_fns[0])
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_anno2()