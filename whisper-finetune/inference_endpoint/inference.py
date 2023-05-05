import urllib.request
import json
import librosa

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
audio = librosa.load("../common_voice_hi_23795238.mp3")[0]
audio = audio.tolist()
data = {"audio": audio}

body = str.encode(json.dumps(data))

# TODO: Replace this with the your REST endpoint link
url = 'https://{endpoint_name}.{endpoint_region}.inference.ml.azure.com/score'
# TODO: Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = ''
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'acpt-ort-whisper-1' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read().decode("utf8", 'ignore')
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
