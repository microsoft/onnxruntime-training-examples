import urllib.request
import json
import librosa

# Request data goes here, the example below assumes JSON formatting.
# More information can be found here: https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
audio = librosa.load("../common_voice_hi_23795238.mp3")[0]
audio = audio.tolist()
data = {"audio": audio}

body = str.encode(json.dumps(data))

endpoint_config = json.load(open("endpoint_config.json"))
endpoint_url = endpoint_config["endpoint_url"]
api_key = endpoint_config["api_key"]

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(endpoint_url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read().decode("utf8", 'ignore')
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
