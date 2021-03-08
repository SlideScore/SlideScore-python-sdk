import json
import sys
import requests
import base64
import string
import re


class SlideScoreErrorException(Exception):
    pass


class SlideScoreResult:
    def __init__(self, dict=None):
        if dict is None:
            self.image_id = 0
            self.image_name = ''
            self.user = None
            self.tma_row = None
            self.tma_col = None
            self.tma_sample_id = None
            self.question = None
            self.answer = None
            return

        self.image_id = int(dict['imageID'])
        self.image_name = dict['imageName']
        self.user = dict['user']
        self.tma_row = int(dict['tmaRow']) if 'tmaRow' in dict else 0
        self.tma_col = int(dict['tmaCol']) if 'tmaCol' in dict else 0
        self.tma_sample_id = dict['tmaSampleID'] if 'tmaSampleID' in dict else ""
        self.question = dict['question']
        self.answer = dict['answer']

        if self.answer[:2] == '[{':
            annos = json.loads(self.answer)
            if len(annos) > 0:
                if hasattr(annos[0], 'type'):
                    self.annotations = annos
                else:
                    self.points = annos
                    
    def toRow(self):
        ret = str(self.image_id) + "\t" + self.image_name + "\t" + self.user + "\t"
        if self.tma_row is not None:
            ret = ret + str(self.tma_row) + "\t" + str(self.tma_col)+"\t" + self.tma_sample_id + "\t"
        ret = ret + self.question + "\t" + self.answer
        return ret


class APIClient(object):
    print_debug = False

    def __init__(self, server, api_token, disable_cert_checking=False):
        if (server[-1] == "/"):
            server = server[:-1]
        self.end_point = "{0}/Api/".format(server)
        self.api_token = api_token
        self.disable_cert_checking = disable_cert_checking

    def perform_request(self, request, data, method="POST"):
        headers = {'Accept': 'application/json'}
        headers['Authorization'] = 'Bearer {auth}'.format(auth=self.api_token)
        url = "{0}{1}".format(self.end_point, request)
        verify=True
        if self.disable_cert_checking:
            verify=False

        if method == "POST":
            response = requests.post(url, verify=verify, headers=headers, data=data)
        else:
            response = requests.get(url, verify=verify, headers=headers, data=data, stream=True)
        if response.status_code != 200:
            response.raise_for_status()

        return response

    def get_images(self, studyid):
        response = self.perform_request("Images", {"studyid": studyid})
        rjson = response.json()
        return rjson

    def get_results(self, studyid, question, email, imageid, caseid):
        response = self.perform_request("Scores", {"studyid": studyid, "question": question, "email": email, "imageid": imageid, "caseid": caseid})
        rjson = response.json()
        return [SlideScoreResult(r) for r in rjson]
        
    def upload_results(self, studyid, results):
        sres = "\n"+"\n".join([r.toRow() for r in results])
        response = self.perform_request("UploadResults", {
                 "studyid": studyid,
                 "results": sres
                 })
        rjson = response.json()
        if (not rjson['success']):
            raise SlideScoreErrorException(rjson['log'])
        return True
        
    def upload_ASAP(self, imageid, user, questions_map, annotation_name, asap_annotation):
        response = self.perform_request("UploadASAPAnnotations", {
                 "imageid": imageid,
                 "questionsMap": '\n'.join(key+";"+value for key, val in questions_map.items()),
                 "user": user,
                 "annotationName": annotation_name,
                 "asapAnnotation": asap_annotation})
        rjson = response.json()
        if (not rjson['success']):
            raise SlideScoreErrorException(rjson['log'])
        return True

    def export_ASAP(self, imageid, user, question):
        response = self.perform_request("ExportASAPAnnotations", {
                 "imageid": imageid,
                 "user": user,
                 "question": question})
        rawresp = response.text
        if rawresp[0] == '<':
            return rawresp
        rjson = response.json()
        if (not rjson['success']):
            raise SlideScoreErrorException(rjson['log'])

    def get_image_server_url(self, imageid):
        response = self.perform_request("GetTileServer?imageId="+str(imageid), None,  method="GET")
        rjson = response.json()
        return ( self.end_point.replace("/Api/","/i/"+str(imageid)+"/"+rjson['urlPart']+"/_files"), rjson['cookiePart'] )

    def _get_filename(self, s):
      fname = re.findall("filename\*?=([^;]+)", s, flags=re.IGNORECASE)
      return fname[0].strip().strip('"')        
        
    def download_slide(self, studyid, imageid, filepath):
        response = self.perform_request("DownloadSlide", {"studyid": studyid, "imageid": imageid}, method="GET")
        fname = self._get_filename(response.headers["Content-Disposition"])
        with open(filepath+'/'+fname, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)

    def get_screenshot_whole(self, imageid, user, question, output_file):
        response = self.perform_request("GetScreenshot", {"imageid": imageid, "withAnnotationForUser": user, "question": question, "level": 11}, method="GET")
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)

