# Slide Score Python SDK

This SDK contains the client library for using API of [Slide Score](https://www.SlideScore.com)	
See the [documentation](https://www.slidescore.com/docs/api/index.html) for more 

Example:

	from slidescore import *

	token="eyJh....."
	url="https://url.slidescore.com/"
	studyid=42

	client = APIClient(url, token)
	
	#download files in a study
	for f in client.get_images(42):
	   print('downloading '+f["name"]+'...', end='', flush=True)
	   client.download_slide(42, f["id"], ".")
	   print('done')
	
	#upload some file
	client.upload_file("C:/pathologyimages/test_slide.tiff", "test", "server_filename.tiff")



