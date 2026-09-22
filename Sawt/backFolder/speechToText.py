import subprocess
import os
import time
import json
from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech


# set the path to the google cloud service account key
google_credentials_path = r"key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
# define the bucket and destination path for the uploaded file
bucket_name = "sawt_bucket"
destination_blob_name = "uploads/audio1.mp3"


def convert_mp4_to_mp3(input_file, output_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(script_dir, "ffmpeg.exe")

    command = [ffmpeg_path, "-i", input_file, "-ar", "16000", "-ac", "1", "-q:a", "0", "-map", "a", output_file, "-y"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError("Conversion in ffmpeg failed")

    return output_file


def upload_to_gcs(bucket_name, destination_blob_name, source_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # upload the file to GCS
    blob.upload_from_filename(source_file_name)
    print(f"\nFile uploaded successfully to: gs://{bucket_name}/{destination_blob_name}")

    return f"gs://{bucket_name}/{destination_blob_name}"


def convert_speech_to_text(gcs_uri):
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        use_enhanced=True,
        language_code="ar-IL",
        alternative_language_codes=[
            "ar-SA",
            "ar-EG",
            "ar-AE",
            "ar-LB",
            "ar-JO",
            "ar-MA",
            "ar-DZ",
            "ar-TN",
            "ar-IQ",
            "en-US",
        ],
        model="latest_long",
    )

    audio = speech.RecognitionAudio(uri=gcs_uri)
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Transcription in progress")

    # wait for transcription to complete
    response = operation.result()
    
    start_time = time.time()
    end_time = time.time()

    # extract each word with timestamps
    words_data = []
    for result in response.results:
        alternative = result.alternatives[0]
        for word_info in alternative.words:
            word_entry = {
                "word": word_info.word,
                "start_time": word_info.start_time.total_seconds(),
                "end_time": word_info.end_time.total_seconds(),
            }
            words_data.append(word_entry)

    return words_data


def save_transcription_to_file(words_data, filename="transcriptionText.json"):
    words_data = sorted(words_data, key=lambda x: x["start_time"])

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(words_data, file, ensure_ascii=False, indent=4)

    print(f"\nWord timestamps saved to {filename}")
    return filename


def process_audio_file(file_path):
    if file_path.endswith(".mp4"):
        mp3_file_path = file_path.replace(".mp4", ".mp3")
        convert_mp4_to_mp3(file_path, mp3_file_path)
        file_path = mp3_file_path

    gcs_uri = upload_to_gcs(bucket_name, destination_blob_name, file_path)
    words_data = convert_speech_to_text(gcs_uri)
    transcription_filename = save_transcription_to_file(words_data)
    return transcription_filename

