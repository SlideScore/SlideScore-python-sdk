DESC = """
Simple script to convert a lot of (big) anno1 entries into anno2's locally. 
Dumps the results to --output after each conversion. 
Since this script requires quite some info about each scorevalue, it uses a full DB as input.
Inspect get_rows if you want to change the selected ScoreValues.

Author: Bart Grosman
"""

import argparse
import sqlite3
import json
import sys
import os
import traceback
import time

import slidescore

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('Email: ') # https://slidescore.com/

def get_rows(db_path: str, itersize=100, size_cutoff = 100 * 1000):
    sql = """SELECT * FROM ScoreValues WHERE Value LIKE '[%' AND LENGTH(Value) > """ + str(int(size_cutoff))

    if not os.path.exists(db_path):
        sys.exit('Input database does not exist: ' + db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row # Get a named return value
    cursor = con.cursor()
    cursor.execute(sql)

    while True:
        batch = cursor.fetchmany(itersize)
        if not batch:
            break
        for result in batch:
            yield result


def get_type(entries):
    anno_type = None
    for entry in entries:
        cur_type = entry['type'] if 'type' in entry else 'points'
        if anno_type is None:
            anno_type = cur_type
        if anno_type != cur_type:
            anno_type = 'mixed'
    return anno_type


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('--input-db', type=str, default='tutim.db',
                        help="""Slide Score database (tutim.db) backup, used to read the relevant ScoreValue entries. Should be the same as the 
                        one connected to using the API!""")
    parser.add_argument('--output', type=str, default='anno2_conversion_log.json',
                        help='Results saved to json file for later analysis')
    parser.add_argument('--upload', action='store_true', help='Whether to upload anno2s to slidescore server, or do a dry run')
    parser.add_argument('--wait', action='store_true', help='Wait for input after every conversion')
    parser.add_argument('--size-cutoff', type=int, default=100 * 1000, help='Size cutoff in bytes, i.e. only convert >100 KB')

    args = parser.parse_args(argv)
    upload = args.upload
    # Create slidescore client
    host = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(host, SLIDESCORE_API_KEY)
    print("Created client for conversion and potential upload")

    results = []
    # Fetch relevant scorevalues to test
    rows = get_rows(args.input_db, size_cutoff=args.size_cutoff)

    for i, row in enumerate(rows):
        result = {
            "success": None,
            "time_needed": None,
            "bytes_before": None,
            "bytes_after": None,
            "error": None,
            "type": None,
            "score_value_id": row["ID"]
        }
        
        ## Try converting to anno2
        try:
            res = dict(row)
            score_value_json = res['Value']
            study_id = row['StudyID']
            image_id = row['ImageID']
            score_id = row['ScoreID']
            tma_core_id = row['TmaCoreID']
            case_id = row['CaseID']
            email = row['Email']

            # Load the anno1
            score_value = json.loads(score_value_json)
            score_value_type = get_type(score_value)

            # Log this row
            res['Value'] = res['Value'][:100]
            print(score_value_type, res)
            
            # Upload anno2
            t0 = time.time()
            local_anno2_path = '/tmp/test_anno2.zip'
            before_bytes = len(score_value_json)
            client.convert_to_anno2(score_value, res, local_anno2_path)
            after_bytes = os.path.getsize(local_anno2_path)
            print("{:.2f} KiB -> {:.2f} KiB, {:.0f}% ".format(before_bytes / 1024, after_bytes / 1024, (after_bytes / before_bytes) * 100))
            sec_needed = time.time() - t0

            if upload:
                # Create DB entry serverside
                # studyId, int? caseId, int imageId, int? tmaCoreId, int? scoreId, string question, string email
                options = { 
                    "studyid": study_id,
                    "imageId": image_id,
                    "scoreId": score_id,
                    "email": email
                }
                if tma_core_id:
                    options['tmaCoreId'] = tma_core_id
                if case_id:
                    options['caseId'] = case_id
                resp = client.perform_request("CreateAnno2", options, method="POST").json()

                # Actually upload the annotation
                client.upload_using_token(local_anno2_path, resp["uploadToken"])
                
                print(f'Uploaded with uuid: {resp["annoUUID"]}')
                print(f'Done, view results at: {SLIDESCORE_HOST}/Image/Details?imageId={image_id}&studyId={study_id}')
            
            result = {
                "success": True,
                "time_needed": sec_needed,
                "bytes_before": before_bytes,
                "bytes_after": after_bytes,
                "error": None,
                "type": score_value_type,
                "score_value_id": row["ID"]
            }
            results.append(result)
        except KeyboardInterrupt:
            print('\nExiting')
            sys.exit(0)
        except Exception as e:
            traceback.print_exc()
            error_message = traceback.format_exc()
            result = {
                "success": False,
                "time_needed": None,
                "bytes_before": None,
                "bytes_after": None,
                "error": error_message,
                "type": None,
                "score_value_id": row["ID"]
            }
            results.append(result)
            print("FAILED", row["ID"])
        
        # Done trying, save results and log some stats
        with open(args.output, 'w') as output_fh:
            json.dump(results, output_fh, indent=2)
        
        mb_saved = sum([r['bytes_before'] - r['bytes_after'] for r in results if r['bytes_after']]) / (1024 * 1024)
        stats = {
            'num_processed': len(results),
            'num_success': len([r for r in results if r['success']]),
            'num_failed': len([r for r in results if not r['success']]),
            'mb_saved': mb_saved
        }
        print(json.dumps(stats), file=sys.stderr)
        if args.wait:
            input('Waiting for keypress... ')

    print(f'Done! Total #: {len(results)}')

if __name__ == "__main__":
    main()