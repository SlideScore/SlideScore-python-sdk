# coding=utf-8
import json
import sys
import requests
import base64
import string
import re
from tusclient import client


class SlideScoreErrorException(Exception):
    pass

class SlideScoreResult:
    """Slidescore wrapper class for storing SlideScore server responses."""
    def __init__(self, dict=None):
        """
        Parameters
        ----------
        slide_dict : dict
            SlideScore server response for annotations/labels.
        """
        if dict is None:
            self.image_id = 0
            self.image_name = ''
            self.case_name = ''
            self.user = None
            self.tma_row = None
            self.tma_col = None
            self.tma_sample_id = None
            self.question = None
            self.answer = None
            return

        self.image_id = int(dict['imageID'])
        self.image_name = dict['imageName']
        self.case_name = dict['caseName']
        self.user = dict['user']
        self.tma_row = int(dict['tmaRow']) if 'tmaRow' in dict else None
        self.tma_col = int(dict['tmaCol']) if 'tmaCol' in dict else None
        self.tma_sample_id = dict['tmaSampleID'] if 'tmaSampleID' in dict else None
        self.question = dict['question']
        self.answer = dict['answer']

        if self.answer is not None and self.answer[:2] == '[{':
            annos = json.loads(self.answer)
            if len(annos) > 0:
                if hasattr(annos[0], 'type'):
                    self.annotations = annos
                else:
                    self.points = annos
                    
    def toRow(self):
        ret = str(self.case_name) + "\t" + str(self.image_id) + "\t" + self.image_name + "\t" + self.user + "\t"
        if self.tma_row is not None:
            ret = ret + str(self.tma_row) + "\t" + str(self.tma_col)+"\t" + self.tma_sample_id + "\t"
        ret = ret + self.question + "\t" + self.answer
        return ret
        
    def __repr__(self):
        return (
            f"SlideScoreResult(case_name={self.case_name}, "
            f"image_id={self.image_id}, "
            f"image_name={self.image_name}, "
            f"user={self.user}, "
            f"tma_row={self.tma_row}, "
            f"tma_col={self.tma_col}, "
            f"tma_sample_id={self.tma_sample_id}, "
            f"question={self.question}, "
            f"answer=length {len(self.answer)})"
        )        


class APIClient(object):
    print_debug = False

    def __init__(self, server, api_token, disable_cert_checking=False):
        """
        Base client class for interfacing with slidescore servers.
        Needs and slidescore_url (example: "https://www.slidescore.com/"), and a api token. Note the ending "/".
        Parameters
        ----------
        server : str
            Path to SlideScore server (without "Api/").
        api_token : str
            API token for this API request.
        disable_cert_checking : bool
            Disable checking of SSL certification (not recommended).
        """    
        if (server[-1] == "/"):
            server = server[:-1]
        self.end_point = "{0}/Api/".format(server)
        self.api_token = api_token
        self.disable_cert_checking = disable_cert_checking

    def perform_request(self, request, data, method="POST", stream=True):
        """
        Base functionality for making requests to slidescore servers. Request should\
        be in the format of the slidescore API: https://www.slidescore.com/docs/api.html
        Parameters
        ----------
        request : str
        data : dict
        method : str
            HTTP request method (POST or GET).
        stream : bool
        Returns
        -------
        Response
        """
        if method not in ["POST", "GET"]:
            raise SlideScoreErrorException(f"Expected method to be either `POST` or `GET`. Got {method}.")
        
        headers = {'Accept': 'application/json'}
        headers['Authorization'] = 'Bearer {auth}'.format(auth=self.api_token)
        url = "{0}{1}".format(self.end_point, request)
        verify=True
        if self.disable_cert_checking:
            verify=False

        if method == "POST":
            response = requests.post(url, verify=verify, headers=headers, data=data)
        else:
            response = requests.get(url, verify=verify, headers=headers, data=data, stream=stream)
        if response.status_code != 200:
            response.raise_for_status()

        return response

    def get_images(self, studyid):
        """
        Get slide data (no slides) for all slides in the study.
        Parameters
        ----------
        studyid : int
        Returns
        -------
        dict
            Dictionary containing the images in the study.
        For example to download all slides in a study with id 1 into the current directory you need to do 
            client = APIClient(url, token)
            for f in client.get_images(1):
                client.download_slide(1, f["id"], ".")

        
        """    
        response = self.perform_request("Images", {"studyid": studyid})
        rjson = response.json()
        return rjson

    def get_cases(self, studyid):
        """
        Get slide data (no slides) for all slides in the study.
        Parameters
        ----------
        studyid : int
        Returns
        -------
        dict
            Dictionary containing the images in the study.
        For example to download all slides in a study with id 1 into the current directory you need to do 
            client = APIClient(url, token)
            for f in client.get_images(1):
                client.download_slide(1, f["id"], ".")

        
        """    
        response = self.perform_request("Images", {"studyid": studyid})
        rjson = response.json()
        return rjson

    def get_studies(self, studyid):
        """
        Get list of studies this token can access
        Parameters
        ----------
        None 
        
        Returns
        -------
        dict
            Dictionary containing the studies.
        """    
        response = self.perform_request("Studies", {})
        rjson = response.json()
        return rjson
        
    def get_results(self, studyid, question=None, email=None, imageid=None, caseid=None):
        """
        Basic functionality to download all answers for a particular study.
        Returns a SlideScoreResult class wrapper containing the information.
        Parameters
        ----------
        study_id : int
            ID of SlideScore study.
        question: string
            Filter for results for this question
        email: string
            Filter for results from this user
        imageid: int
            Filter for results on this image
        caseid: int
            Filter for results on this case
        Returns
        -------
        List[SlideScoreResult]
            List of SlideScore results.
        """
        response = self.perform_request("Scores", {"studyid": studyid, "question": question, "email": email, "imageid": imageid, "caseid": caseid})
        rjson = response.json()
        return [SlideScoreResult(r) for r in rjson]
        
    def get_config(self, study_id):
        """
        Get the configuration of a particular study. Returns a dictionary.
        Parameters
        ----------
        study_id : int
            ID of SlideScore study.
        Returns
        -------
        dict
        """
        response = self.perform_request("GetConfig", {"studyid": study_id})
        rjson = response.json()

        if not rjson["success"]:
            raise SlideScoreErrorException(f"Configuration for study id {study_id} not returned succesfully")

        return rjson["config"]        
        
    def upload_results(self, studyid, results):
        """
        Basic functionality to upload results/answers made for a particular study.
        Returns true if successful.
        results should be a list of strings, where each elemement is a line of text of the following format:
        imageID - tab - imageNumber - tab - author - tab - question - tab - answer
        
        Parameters
        ----------
        studyid : int
        results : List[str]
        Returns
        -------
        bool
        """    
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
        """
        Returns the image server slidescore url for given image.
        Parameters
        ----------
        image_id : int
            SlideScore Image ID.
        Returns
        -------
        tuple
            Pair consisting of url, cookie.
        """
        if self.base_url is None:
            raise RuntimeError
        response = self.perform_request("GetTileServer?imageId="+str(imageid), None,  method="GET")
        rjson = response.json()
        return ( 
            self.end_point.replace("/Api/","/i/"+str(imageid)+"/"+rjson['urlPart']+"/_files"), 
            rjson['cookiePart'] 
        )

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

    def request_upload(self, destination_folder, destination_filename, studyid):
        response = self.perform_request("RequestUpload", {"filename": destination_filename, "folder": destination_folder, "studyId": studyid}, method="POST")
        if response.text[0] == '"':
            raise SlideScoreErrorException("Failed requesting upload: " + response.text);
        return response.json()['token']

    def finish_upload(self, upload_token, upload_url):
        fileid=upload_url[upload_url.rindex('/')+1::]
        response = self.perform_request("FinishUpload", {"id": fileid, "token": upload_token}, method="POST")
        if response.text != '"OK"':
            raise SlideScoreErrorException("Failed finishing upload: " + response.text);
         
    def upload_file(self, source_filename, destination_path, destination_filename):
        """
        Upload a file to the server.
        Parameters
        ----------
        source_filename: string
            Local path to the file to upload
        destination_path: string
            path (without filename) on the server
        destination_filename: string 
            filename to use on the server
        
        """
        uploadToken = self.request_upload(destination_path, destination_filename, None)
        uploadClient = client.TusClient(self.end_point.replace('/Api/','/files/'))
        uploader = uploadClient.uploader(source_filename, chunk_size=10*1000*1000, metadata={'uploadtoken': uploadToken, 'apitoken': self.api_token})
        # Uploads the entire file.
        # This uploads chunk by chunk.
        uploader.upload()
        self.finish_upload(uploadToken, uploader.url)

    def add_slide(self, study_id, destination_filename):
        
        response = self.perform_request("AddSlide", {"studyId": study_id, "path": destination_filename}, method="POST")
        if response.text != '"OK"':
            raise SlideScoreErrorException("Failed adding slide: " + response.text);
        
        
    
    
