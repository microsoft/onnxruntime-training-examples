# MyVoice
MyVoice uses onnxruntime iOS on device training APIs to perform voice verification on client without transmitting any user voice data.

The application uses the user audio recordings and sample audio recordings from [librispeech](https://huggingface.co/datasets/librispeech_asr) dataset to train the model to identify user voice. 



![Screenshot](./screenshot.png)
## Model information

MyVoice uses [wav2vec model](https://huggingface.co/superb/wav2vec2-base-superb-sid) trained on VoiceCeleb database and applies transfer learning principles to create a model that can confidently identify user voice from 1 minute of user audio.

## Set up

### Install the Pod dependencies

From this directory, run:

```bash
pod install
```

### Generate Artifacts
[Artifacts generation](./artifacts_gen.ipynb) script downloads the model from huggingface and creates the folder named `artifacts` containing training artifacts. 

Next, open MyVoice.xcworkspace file in Xcode and import training artifacts by right clicking on artifacts directory and selecting "Add Files to "MyVoice"".


### Download Sample Recordings
To download sample recordings from dataset, run [Recording generation](./recording_gen.ipynb) script. The script downloads recordings sample in recording directory. 

Next, open MyVoice.xcworkspace file in Xcode and import sample audio files by right clicking on recording directory and selecting "Add Files to "MyVoice"".

## Build and run

Open the generated SpeechRecognition.xcworkspace file in Xcode to build and run the example.