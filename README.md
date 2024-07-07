# Slide Score Python SDK

This SDK contains the client library for using API of [Slide Score](https://www.SlideScore.com)	
See the [documentation](https://www.slidescore.com/docs/api/index.html) for more 

# Examples

For more examples see the examples folder

## Basic usage

- Import the module
- setup token
- setup URL of the Slide Score server
- create an instance of the API Client

    from slidescore import *


    token="eyJ....<your token>...."
    url="https://slidescore.example.com/"

    client = APIClient(url, token)
    
## Downloading a slide 

Downloads a slide to the current directory ("."). Check the URL of the slide for image ID and study ID, or click Export cases button on study overview to get a list of slide IDs.

    studyid=1
    imageid=2

    client.download_slide(studyid, imageid, ".")


## Uploading and adding a slide to a study

    localFilePath="C:/file_to_upload.tiff"
    uploadFolder="UploadTest"
    serverFileName="renamedSlide.tiff"

    client.upload_file(localFilePath, uploadFolder, serverFileName)
    client.add_slide(studyid, uploadFolder+"/"+serverFileName) 
    
## Upload answers - study results

Results are uploaded in the same format as the download.

    resultsFilePath="c:/Users/User/Downloads/Study_23_06_21_11.txt"
    with open(resultsFilePath, "r") as f:
        res = f.read()
    client.upload_results(studyid, res)


## Set slide description

You can use (limited) set of HTML tags in slide, case, study and module descriptions:

    client.update_slide_description(studyid, imageid, 'Carina Nebula: Cosmic Cliffs, Glittering Landscape of Star Birth. Image Credit: NASA, ESA, CSA, and STScI <a href="https://esawebb.org/news/weic2205/">Original</a>');


## Make a request directly

The SDK doesn't include methods for all possible calls, sometimes you need to make the API request yourself:

    response=clientlocal.perform_request("UpdateSlideName", {"imageId":imageid, "newName":'renamedSlide'}, method="POST")
    rjson = response.json()
    if 'success' not in rjson or rjson['success'] != True:
        raise SlideScoreErrorException("Failed updating slide name: " + response.text);


