# coding=utf-8
import json
import sys
import requests
import base64
import string
import re
import os
import datetime
from tusclient import client

from .lib.utils import read_slidescore_json
from .lib.Encoder import Encoder
from .lib.AnnoClasses import Points, Polygons

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
            self.id = 0
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

        self.id = int(dict['id'])
        self.image_id = int(dict['imageID'])
        self.image_name = dict['imageName']
        self.case_name = dict['caseName'] if 'caseName' in dict else None
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
        """
        Options:
            1. ImageID	Name	By	Question	Answer = 5
            2. ImageID	Name	By	TMA Row	TMA Col	TMA Sample	Question	Answer = 8
            3. CaseName	ImageID	Name	By	Question	Answer = 6
            4. CaseName	ImageID	Name	By	TMA Row	TMA Col	TMA Sample	Question	Answer = 9
            5. CaseName	By	Question	Answer = 4
        """
        answer_str = ''
        
        # Prepend case name if needed
        if self.case_name is not None:
            answer_str += f'{self.case_name}	'
        
        # Early return option 5
        if self.image_id is None:
            answer_str += f'{self.case_name}	{self.user}	{self.question}	{self.answer}'
            return answer_str
        
        # Add image_id, name and user
        answer_str += f'{self.image_id}	{self.image_name}	{self.user}	'
        # If TMA is defined, add it
        if self.tma_row is not None:
            answer_str += f'{self.tma_row}	{self.tma_col}	{self.tma_sample_id}	'
        # Finally add the question and answer
        answer_str += f'{self.question}	{self.answer}'
        return answer_str

        
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

class SlideScoreSession:
    """Wrapper class for storing SlideScore tracking sessions"""
    def __init__(self, dict=None):
        if dict is None:
            self.id = 0
            self.image_id = 0
            self.email = ''
            self.length = ''
            self.study_id = None
            self.created_on = None
            return

        self.id = int(dict['id'])
        self.image_id = int(dict['imageID'])
        self.email = dict['email']
        self.length = int(dict['length'])
        self.study_id = int(dict['studyID'])
        #ignore milliseconds, sometimes there are 7 digits
        self.created_on = datetime.datetime.strptime(dict["createdOn"][:19],"%Y-%m-%dT%H:%M:%S")
        
class SlideScoreSessionEvent:
    """Wrapper class for storing SlideScore tracking session events"""
    def __init__(self, s=None):
        if s is None:
            self.timestamp = 0
            self.x = 0
            self.y = 0
            self.width = 0
            self.height = 0
            self.cursor_x = 0
            self.cursor_y = 0
            return
        terms=s.split('\t')
        self.timestamp = int(terms[0])
        self.x = int(terms[1])
        self.y = int(terms[2])
        self.width = int(terms[3])
        self.height = int(terms[4])
        self.cursor_x = int(terms[5])
        self.cursor_y = int(terms[6])

        

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

    def perform_request(self, request, data, method="GET", stream=True):
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
        Get all case names and IDs
        Parameters
        ----------
        studyid : int
        Returns
        -------
        dict
            Dictionary containing the cases in the study.
        For example:
            client = APIClient(url, token)
            for c in client.get_cases(1):
                print(str(c["id"])+" - " + c["name"])

        
        """    
        response = self.perform_request("Cases", {"studyid": studyid})
        rjson = response.json()
        return rjson

    def get_studies(self):
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

    def get_config_files(self, study_id):
        """
        Get the configuration files of a particular study. Returns a dictionary with file contents for each file.
        Parameters
        ----------
        study_id : int
            ID of SlideScore study.
        Returns
        -------
        dict
        """
        response = self.perform_request("GetConfigFiles", {"studyid": study_id})
        rjson = response.json()

        if not rjson["success"]:
            raise SlideScoreErrorException(f"Configuration for study id {study_id} not returned succesfully")

        return rjson        
        
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
                 }, method="POST")
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
                 "asapAnnotation": asap_annotation}
                 , method="POST")
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
      fname = re.findall("filename*?=([^;]+)", s, flags=re.IGNORECASE)
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
         
    def upload_file(self, source_filename, destination_path, destination_filename=None):
        """
        Upload a file to the server.
        Parameters
        ----------
        source_filename: string
            Local path to the file to upload
        destination_path: string
            path (without filename) on the server
        destination_filename: string 
            filename to use on the server, None to use the source filename
        
        """
        if destination_filename==None:
            destination_filename = os.path.basename(source_filename)
        uploadToken = self.request_upload(destination_path, destination_filename, None)
        self.upload_using_token(source_filename, uploadToken)

    def upload_using_token(self, source_filename, upload_token):
        uploader_endpoint = self.end_point.replace('/Api/','/files/')
        uploadClient = client.TusClient(uploader_endpoint)
        uploader = uploadClient.uploader(source_filename, chunk_size=10*1000*1000, metadata={'uploadtoken': upload_token, 'apitoken': self.api_token})
        # Uploads the entire file.
        # This uploads chunk by chunk.
        uploader.upload()
        self.finish_upload(upload_token, uploader.url)

    def add_slide(self, study_id, destination_filename):
        
        response = self.perform_request("AddSlide", {"studyId": study_id, "path": destination_filename}, method="POST")
        rjson=response.json()
        if response.text[0] == '"':
            raise SlideScoreErrorException("Failed adding slide: " + rjson);
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed adding slide: " + response.text);
        return { "id": rjson['id'], "isOOF": rjson["isOOF"]}

    def reimport(self, study_name):
        response = self.perform_request("Reimport", {"studyName": study_name}, method="POST")
        if response.text[0] == '"':
            raise SlideScoreErrorException("Failed reimporting: " + response.text);
        rjson=response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed reimporting: " + rjson["log"]);
        return { "id": rjson['id'], "log": rjson["log"]}
        
    def get_slide_path(self, image_id):
        response = self.perform_request("GetSlidePath", {"imageId": image_id}, method="GET")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed getting slide path: " + response.text);
        return rjson['path'] 
        
    def get_slide_description(self, image_id):
        response = self.perform_request("GetSlideDescription", {"imageId": image_id}, method="GET")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed getting slide description: " + response.text);
        return rjson['description'] 
        
    def get_case_description(self, case_id):
        response = self.perform_request("GetCaseDescription", {"caseId": case_id}, method="GET")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed getting case description: " + response.text);
        return rjson['description'] 
        

    def update_slide_path(self, image_id, new_path):
        response = self.perform_request("UpdateSlidePath", {"imageId": image_id, "newPath": new_path}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed updating slide path: " + response.text);

    def update_slide_description(self, study_id, image_id, new_description):
        response = self.perform_request("SetSlideDescription", {"imageId": image_id, "studyId": study_id, "description": new_description}, method="POST")
        rjson = response.json()
        if response.text != '{}':
            raise SlideScoreErrorException("Failed updating slide description: " + response.text);
    
    def update_slide_name(self, image_id, new_name):
        response = self.perform_request("UpdateSlideName", {"imageId": image_id, "newName": new_name}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed updating slide name: " + response.text);

    def add_question(self, study_id, question_spec):
        response = self.perform_request("AddQuestion", {"studyId": study_id, "questionSpec": question_spec}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed adding question: " + response.text);
        return rjson["id"]

    def update_question(self, study_id, score_id, order, question_spec):
        response = self.perform_request("UpdateQuestion", {"studyId": study_id, "scoreId": score_id, "order": order, "questionSpec": question_spec}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed updating question: " + response.text);
        return rjson["id"]

    def remove_question(self, study_id, score_id):
        response = self.perform_request("RemoveQuestion", {"studyId": study_id, "scoreId": score_id}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed removing question: " + response.text);
        return True
    
    def set_slide_tma_map(self, study_id, image_id, tma_map_name):
        response = self.perform_request("SetSlideTMAMap", {"studyId": study_id, "imageId": image_id, "tmaMapName": tma_map_name}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed updating slide TMA: " + response.text);

    def create_tma_map(self, study_id, tma_map_filename):
        response = self.perform_request("CreateTMAMap", {"studyId": study_id, "tmaMapFileName": tma_map_filename}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed creating TMA map: " + response.text);
        return rjson["mapName"]
        
    def is_slide_out_of_focus(self, study_id, image_id):
        response = self.perform_request("IsSlideOutOfFocus", {"studyId": study_id, "imageId": image_id}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed checking out-of-focus: " + response.text);
        return rjson["isOOF"]

    def get_raw_tile(self, study_id, image_id, level, x, y, width, height,
            jpeg_quality):
        """
        Get slide pixels
        Parameters
        ----------
        study_id : int
        image_id : int
        level: int 
            Level in the slide (0-highest detail), based on slide metadata
        x: int
        y: int
            X and Y of the tile
        width: int
        height: int
            Size of the requested region
        jpeg_quality: int
        
        Returns
        -------
        jpeg file

        
        """    
        response = self.perform_request("GetRawTile", {"studyid": study_id, "imageId": image_id,
        "x": x, "y": y, "width": width, "height": height, "level": level, "jpegQuality": jpeg_quality})
        return response        

        
    def convert_to_anno2(self, items, metadata, output_path):
        """Converts a SlideScore Annotation Object to the new Anno2 zip based format
        Only supports annotations of points or polygons/brush. Will error otherwise.

        anno1_data: Dictionary containing the annotation like: [{"type": "brush", "positivePolygons": [] ...]
        metadata: Dictionary containing any metadata regarding the annotation, will be included as JSON in output
        output_path: string of the path on disk the anno2.zip will be written to
        """
        # Allow pre-loaded Points and Polygons objects
        if not isinstance(items, Points) and not isinstance(items, Polygons):
            items = read_slidescore_json(items)
        
        encoder = Encoder(items, big_polygon_size_cutoff=100 * 100)
        encoder.generate_tile_data(256)        
        encoder.populate_lookup_tables()
        
        if metadata:
            encoder.add_metadata(metadata)

        encoder.dump_to_file(output_path)
        print('Done converting to anno2')


    def convert_annotation_to_anno2(self, study_id, case_id, image_id, tma_core_id,
            score_id, question, email, metadata):
        """Converts an existing annotation answer to the new Anno2 zip based format
        case_id, tma_core_id, score_id, and question are optional
        metadata must be a JSON object for example "{}"
        You have to specify one of score_id and question
        """
        
        response = self.perform_request("ConvertAnnotationToAnno2", {"studyId": study_id, "caseId": case_id, "imageId": image_id, 
                "tmaCoreId": tma_core_id, "scoreId": score_id, "question": question, "email": email, "metadata": metadata}, 
                method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed converting: " + response.text);
        return rjson["annoUUID"]

    def create_anno2(self, study_id, case_id, image_id, tma_core_id,
            score_id, question, email):
        """Creates a new Anno2 format record
        Returns object with  "uploadToken": "FEU...." and "annoUUID": "8f51008c-9ede-e8b4-cba8-55a2cf6c73bf",
        
        You have to specify one of score_id and question
        """
        
        response = self.perform_request("CreateAnno2", {"studyId": study_id, "caseId": case_id, "imageId": image_id, 
                "tmaCoreId": tma_core_id, "scoreId": score_id, "question": question, "email": email}, 
                method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed creating: " + response.text);
        return rjson

        
    def generate_login_link(self, username, expires_on):
        response = self.perform_request("GenerateLoginLink", {"username": username, "expiresOn": expires_on}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed generating link: " + response.text);
        return rjson["link"]

    def generate_student_account(self, username, email, class_id):
        response = self.perform_request("GenerateStudentAccount", {"username": username, "email": email, "classId": class_id}, method="POST")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed generating account: " + response.text);
        return rjson["password"]

    def get_sessions(self, studyid, email=None, imageid=None):
        response = self.perform_request("Sessions", {"studyId": studyid, "imageId": imageid, "email": email})
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed getting sessions: " + response.text);
        return [SlideScoreSession(r) for r in rjson["sessions"]]
        
    def get_session_events(self, studyid, sessionid):
        response = self.perform_request("SessionEvents", {"studyId": studyid, "sessionId": sessionid})
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException("Failed getting session events: " + response.text);
        return [SlideScoreSessionEvent(r) for r in rjson["events"]]