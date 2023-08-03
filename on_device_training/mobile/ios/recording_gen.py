from datasets import load_dataset
import numpy as np
import os
from scipy.io import wavfile

# Load the dataset 
num_examples = 20
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True).take(num_examples)

# process the audio data from the dataset so that it is the same length (10 seconds)
max_len = 160000
def preprocess(audio_array):
    # trim the audio to max_len samples if it is longer than that
    if len(audio_array) > max_len:
        return audio_array[:max_len].astype(np.float32)

    else:
        # pad the shorter arrays with zeros to match the length of the longest array
        pad_len = max_len - len(audio_array)
        return np.pad(audio_array, (0, pad_len), mode='constant').astype(np.float32)


audio_data = [preprocess(example["audio"]["array"]) for example in dataset]

def export_to_wav_folder(audio_list, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, audio_data in enumerate(audio_list):
        # Normalize the audio data to the appropriate range for 16-bit WAV files (-32768 to 32767)
        normalized_audio = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        output_path = os.path.join(output_folder, f"other_{i}.wav")

        # Write the WAV file
        wavfile.write(output_path, 16000, normalized_audio) 

export_to_wav_folder(audio_data, "./MyVoice/recordings")


