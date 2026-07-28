# Slide Score Python SDK

This SDK contains the client library for using API of [Slide Score](https://www.SlideScore.com)	
See the [documentation](https://www.slidescore.com/docs/api/index.html) for more 

# Examples

For more examples see the examples folder

## Basic usage

Import the module and use the token and Slide Score server URL to create an instance of the API Client:

    from slidescore import *
    api_key =  "eyb..<your api token>"
    slidescore_host =  'https://example.slidescore.com/'

    client = APIClient(slidescore_host, api_key)

## Downloading a slide 

Downloads a slide to the current directory ("."). Check the URL of the slide for image ID and study ID, or click Export cases button on study overview to get a list of slide IDs.

    study_id=1
    image_id=2

    client.download_slide(study_id, image_id, ".")


## Uploading and adding a slide to a study


    local_file_path = "/tmp/file_to_upload.tiff"
    server_upload_folder = "UploadTest"
    server_file_name = "renamedSlide.tiff"
    study_id = 1

    client.upload_file(
        local_file_path,
        server_upload_folder,
        server_file_name,
    )

    client.add_slide(
        study_id,
        server_upload_folder + "/" + server_file_name,
    )    
    
## Upload answers - study results

Results are uploaded in the same format as the download.

    results_file_path="c:/Users/User/Downloads/Study_23_06_21_11.txt"
    with open(results_file_path, "r") as f:
        res = f.read()
    client.upload_results(study_id, res)


## Set slide description

You can use (limited) set of HTML tags in slide, case, study and module descriptions:

    client.update_slide_description(study_id, image_id, 'Carina Nebula: Cosmic Cliffs, Glittering Landscape of Star Birth. Image Credit: NASA, ESA, CSA, and STScI <a href="https://esawebb.org/news/weic2205/">Original</a>');


## Make a request directly

The SDK doesn't include methods for all possible calls, sometimes you need to make the API request yourself:

    response=clientlocal.perform_request("UpdateSlideName", {"imageId":image_id, "newName":'renamedSlide'}, method="POST")
    rjson = response.json()
    if 'success' not in rjson or rjson['success'] != True:
        raise SlideScoreErrorException("Failed updating slide name: " + response.text);


