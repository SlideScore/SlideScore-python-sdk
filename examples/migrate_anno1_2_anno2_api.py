DESC = """
Simple script to convert a lot of (big) anno1 entries into anno2's using the API. 
Dumps the results to --output after each conversion. 

You can get the relevant Anno1 values with: "SELECT ID FROM ScoreValues WHERE Value LIKE '[%' OR Value LIKE '{%' AND LENGTH(Value) > 100 * 1000;"

Author: Bart Grosman
"""
import argparse
import json
import sys
import os
import datetime
import traceback

import slidescore

# Either set the environment variables, or hardcode your settings below
SLIDESCORE_API_KEY = os.getenv('SLIDESCORE_API_KEY') or input('What is your Slidescore API key: ') # eyb..
SLIDESCORE_HOST = os.getenv('SLIDESCORE_HOST') or input('What is your Slidescore host: ') # https://slidescore.com/
SLIDESCORE_EMAIL = os.getenv('SLIDESCORE_EMAIL') or input('Email: ') # https://slidescore.com/

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument('--input', type=str, default='score_value_ids.tsv',
                        help="""List of Score Value IDs that should get converted into anno2s, 1 id per line.""")
    parser.add_argument('--output', type=str, default='anno2_conversion_log.json',
                        help='Results saved to json file for later analysis')

    args = parser.parse_args()
    
    # Check the login credentials and create an API client
    if not SLIDESCORE_API_KEY or not SLIDESCORE_HOST:
        sys.exit('SLIDESCORE_API_KEY or SLIDESCORE_HOST not set, please set these variables for your setup')
    # Remove "/" suffix if needed
    SLIDESCORE_HOST = SLIDESCORE_HOST[:-1] if SLIDESCORE_HOST.endswith('/') else SLIDESCORE_HOST

    client = slidescore.APIClient(SLIDESCORE_HOST, SLIDESCORE_API_KEY)
    print('Created API client')

    with open(args.input, 'r') as score_value_ids:
        results = []
        for id in score_value_ids:
            id = id.strip()
            if len(id) == 0:
                continue
            try:
                metadata = json.dumps({
                    "svid": id,
                    "data_converted": datetime.datetime.now().isoformat()
                })
                resp = client.perform_request("ConvertScoreValueToAnno2", {"scoreValueId": int(id), "metadata": metadata}, method="POST")
                if resp.status_code != 200:
                    result = {
                        "id": id,
                        "success": False,
                        "msg": 'non 200 status code: ' + resp.status_code
                    }
                    results.append(result)
                else:
                    resp_data = resp.json()
                    if resp_data['success'] != True:
                        result = {
                            "id": id,
                            "success": False,
                            "msg": 'Non success: ' + resp_data['error']
                        }
                        results.append(result)
                    else:
                        # Success!
                        result = {
                            "id": id,
                            "success": True,
                            "msg": 'Anno2ID=' + resp_data['annoUUID']
                        }
                        results.append(result)
            except KeyboardInterrupt:
                print('\nExiting')
                sys.exit(0)
            except:
                error_message = traceback.format_exc()
                result = {
                    "id": id,
                    "success": False,
                    "msg": error_message
                }
                results.append(result)

            with open(args.output, 'w') as output_fh:
                json.dump(results, output_fh, indent=2)
            
            stats = {
                'num_processed': len(results),
                'num_success': len([r for r in results if r['success']]),
                'num_failed': len([r for r in results if not r['success']]),
            }
            print(json.dumps(stats), file=sys.stderr)

    print(f'Done')