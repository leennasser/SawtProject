import os
import shutil
import json
import string
import hashlib
import re
import subprocess
import whisper
from langdetect import detect
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from flask import (
    Flask,
    request,
    Response,
    send_from_directory,
    render_template,
    session,
    redirect,
    url_for,
    abort,
    jsonify,
)
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from jinja2 import Environment, FileSystemLoader
from speechToText import process_audio_file
from audioSubsystem import (
    summarize_audio_podcast,
    save_audio_info,
    extract_and_save_audio_clips,
)
from videoSubsystem import (
    summarize_visual_content_podcast,
    save_visual_content_info,
    extract_and_save_visual_content_clips,
)
from mergeSubsystem import merge_highlights_for_video_podcast


app = Flask(
    __name__,
    static_folder=os.path.join(os.getcwd(), "assetsFolder"),
    static_url_path="/assetsFolder",
)


pages_english_folder = os.path.join(os.getcwd(), "pagesEnglish")
pages_arabic_folder = os.path.join(os.getcwd(), "pagesArabic")


data_folder = "podcastDataFolder"
os.makedirs(data_folder, exist_ok=True)


@app.route("/assetsFolder/<path:filename>")
def serve_static_files(filename):
    return send_from_directory("assetsFolder", filename)


# function for correct language folder


def render_custom_template(template_name, lang="english", **context):
    folder = pages_arabic_folder if lang == "arabic" else pages_english_folder
    env = Environment(loader=FileSystemLoader(folder))
    template = env.get_template(template_name)
    return template.render(**context)


# function to save podcast data to DB


def save_podcast_data(
    email,
    filename,
    podcast_folder,
    transcription_path,
    audio_info_path=None,
    audio_clips_folder=None,
    visual_info_path=None,
    visual_clips_folder=None,
    video_info_path=None,
    video_clips_folder=None,
):
    # load transcription
    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription_data = json.load(f)

    # load audio info
    audio_info_data = None
    if audio_info_path:
        with open(audio_info_path, "r", encoding="utf-8") as f:
            audio_info_data = json.load(f)

    # load visual info
    visual_info_data = None
    if visual_info_path:
        with open(visual_info_path, "r", encoding="utf-8") as f:
            visual_info_data = json.load(f)

    # load video info
    video_info_data = None
    if video_info_path:
        with open(video_info_path, "r", encoding="utf-8") as f:
            video_info_data = json.load(f)

    # collect audio clips
    audio_clips = []
    if audio_clips_folder and os.path.exists(audio_clips_folder):
        for fname in os.listdir(audio_clips_folder):
            if fname.lower().endswith((".mp3", ".wav")):
                audio_clips.append({"clip_path": os.path.join(audio_clips_folder, fname)})

    # collect visual clips
    visual_clips = []
    if visual_clips_folder and os.path.exists(visual_clips_folder):
        for fname in os.listdir(visual_clips_folder):
            if fname.endswith(".mp4"):
                visual_clips.append({"clip_path": os.path.join(visual_clips_folder, fname)})

    # collect video clips
    video_clips = []
    if video_clips_folder and os.path.exists(video_clips_folder):
        for fname in os.listdir(video_clips_folder):
            if fname.endswith(".mp4"):
                video_clips.append({"clip_path": os.path.join(video_clips_folder, fname)})

    # final document
    podcast_document = {
        "email": email,
        "filename": filename,
        "podcast_folder": podcast_folder,
        "transcription": transcription_data,
        "audio_info": audio_info_data,
        "visual_info": visual_info_data,
        "video_info": video_info_data,
        "audio_clips": audio_clips,
        "visual_clips": visual_clips,
        "video_clips": video_clips,
    }

    podcasts_collection.insert_one(podcast_document)


# functions for upload page 


def is_arabic_language(uploaded_file_path):
    temp_extracted_audio = None
    temp_sample_clip = "tempSample.wav"

    try:
        ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
        
        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(f"The ffmpeg not found at: {ffmpeg_path}")
        
        AudioSegment.converter = ffmpeg_path
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()
        video_formats = [".mp4", ".mov", ".avi", ".mkv"]

        if file_extension in video_formats:
            temp_extracted_audio = "tempExtractedAudio.wav"
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    uploaded_file_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    temp_extracted_audio,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio_input_path = temp_extracted_audio
        else:
            audio_input_path = uploaded_file_path

        # extract first 10 seconds of audio
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                audio_input_path,
                "-t",
                "10",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                temp_sample_clip,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # whisper model transcription
        whisper_model = whisper.load_model("base")
        transcription_result = whisper_model.transcribe(temp_sample_clip)
        transcribed_text = transcription_result["text"].strip()

        if len(transcribed_text) < 5:
            print("Little transcribed text to detect language")
            return False

        detected_language = detect(transcribed_text)
        print("Detected language:", detected_language)

        return detected_language == "ar" or bool(re.search(r"[\u0600-\u06FF]", transcribed_text))

    except Exception as error:
        print("Language detection failed")
        return False

    finally:
        if temp_extracted_audio and os.path.exists(temp_extracted_audio):
            os.remove(temp_extracted_audio)
        if os.path.exists(temp_sample_clip):
            os.remove(temp_sample_clip)


def get_file_checksum(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_all_checksums_for_user(user_folder):
    checksums = []
    if not os.path.exists(user_folder):
        return checksums
    for podcast_folder in os.listdir(user_folder):
        podcast_path = os.path.join(user_folder, podcast_folder)
        checksum_file = os.path.join(podcast_path, "checksum.txt")
        if os.path.isfile(checksum_file):
            with open(checksum_file, "r") as f:
                checksums.append(f.read().strip())
    return checksums


def is_duplicate_podcast(checksum, user_folder):
    existing_checksums = get_all_checksums_for_user(user_folder)
    return checksum in existing_checksums


def save_checksum(checksum, podcast_folder):
    checksum_file = os.path.join(podcast_folder, "checksum.txt")
    with open(checksum_file, "w") as f:
        f.write(checksum)


def next_podcast_folder(user_folder):
    existing = [f for f in os.listdir(user_folder) if f.startswith("podcast")]
    numbers = []
    for folder in existing:
        try:
            num = int(folder.replace("podcast", ""))
            numbers.append(num)
        except ValueError:
            continue
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(user_folder, f"podcast{next_num}")


def save_data_to_folder(src, data_folder):
    # if the source is a file
    if os.path.isfile(src):
        destination_file_path = os.path.join(data_folder, os.path.basename(src))
        shutil.move(src, destination_file_path)
        return destination_file_path
    # if the source is a directory
    elif os.path.isdir(src):
        destination_folder_path = os.path.join(data_folder, os.path.basename(src))
        if os.path.exists(destination_folder_path):
            shutil.rmtree(destination_folder_path)
        shutil.move(src, destination_folder_path)
        return destination_folder_path


# this functions for run transcription and visual content analysis concurrently then use them in upload mp4 podcast process


def process_transcription(file_path, podcast_folder):
    transcription_path = process_audio_file(file_path)
    if not transcription_path.startswith(podcast_folder):
        transcription_path = save_data_to_folder(transcription_path, podcast_folder)
    return summarize_audio_podcast(file_path, transcription_path), transcription_path


def process_visual(file_path, podcast_folder):
    visual_content_highlights = summarize_visual_content_podcast(file_path)
    return visual_content_highlights


# functions for dashboard page


def load_audio_info():
    email = session.get("email")
    if not email:
        return {}
    user_folder = os.path.join(data_folder, email)
    podcasts = []
    if os.path.exists(user_folder):
        for podcast_dir in os.listdir(user_folder):
            podcast_path = os.path.join(user_folder, podcast_dir)
            if os.path.isdir(podcast_path):
                info_path = os.path.join(podcast_path, "audioInfo.json")
                if os.path.exists(info_path) and os.path.getsize(info_path) > 0:
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        info["podcast_dir"] = podcast_dir
                        podcasts.append(info)
    return {"podcasts": podcasts}


def load_video_info():
    email = session.get("email")
    if not email:
        return {}
    user_folder = os.path.join(data_folder, email)
    podcasts = []
    if os.path.exists(user_folder):
        for podcast_dir in os.listdir(user_folder):
            podcast_path = os.path.join(user_folder, podcast_dir)
            if os.path.isdir(podcast_path):
                info_path = os.path.join(podcast_path, "videoInfo.json")
                if os.path.exists(info_path) and os.path.getsize(info_path) > 0:
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        info["podcast_dir"] = podcast_dir
                        podcasts.append(info)
    return {"podcasts": podcasts}


def natural_sort_key(filename):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", filename)]


# routes


@app.route("/")
@app.route("/indexEnglish")
def index_english():
    return render_custom_template("indexEnglish.html", lang="english")


@app.route("/indexArabic")
def index_arabic():
    return render_custom_template("indexArabic.html", lang="arabic")


@app.route("/logoutEnglish")
def logout_english():
    session.clear()
    return redirect(url_for("index_english"))


@app.route("/logoutArabic")
def logout_arabic():
    session.clear()
    return redirect(url_for("index_arabic"))


@app.route("/uploadEnglish", methods=["GET", "POST"])
def upload_page_english():
    if request.method == "GET":
        return render_custom_template("uploadEnglish.html")

    if "podcast-file-input" not in request.files:
        return "No file part", 400

    email = session.get("email")
    if not email:
        return redirect(url_for("login_english"))

    file = request.files["podcast-file-input"]
    if file.filename == "":
        return "No selected file", 400

    # create user folder
    user_folder = os.path.join(data_folder, email)
    os.makedirs(user_folder, exist_ok=True)

    # save temporarily to compute checksum
    temp_path = os.path.join(user_folder, file.filename)
    file.save(temp_path)

    # check if podcast is Arabic language
    if not is_arabic_language(temp_path):
        os.remove(temp_path)
        return "Arabic podcasts only", 400

    checksum = get_file_checksum(temp_path)
    if is_duplicate_podcast(checksum, user_folder):
        os.remove(temp_path)
        return "Already been uploaded", 400

    # create new podcast folder
    podcast_folder = next_podcast_folder(user_folder)
    os.makedirs(podcast_folder)

    # move uploaded file into podcast folder
    file_path = os.path.join(podcast_folder, file.filename)
    shutil.move(temp_path, file_path)

    # save checksum
    save_checksum(checksum, podcast_folder)

    # determine file extension
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    if ext == ".mp3":
        # audio data
        transcription_path = process_audio_file(file_path)
        if not transcription_path.startswith(podcast_folder):
            transcription_path = save_data_to_folder(transcription_path, podcast_folder)

        key_statements, silent_sections, summarized_statements, kmeans = (summarize_audio_podcast(file_path, transcription_path))

        if key_statements:
            output_audio_info_file = save_audio_info(os.path.basename(file_path), key_statements, summarized_statements)
            output_audio_info_file = save_data_to_folder(output_audio_info_file, podcast_folder)

            output_audio_clips_folder = extract_and_save_audio_clips(file_path, key_statements)
            output_audio_clips_folder = save_data_to_folder(output_audio_clips_folder, podcast_folder)

            # save podcast data
            save_podcast_data(
                email=email,
                filename=file.filename,
                podcast_folder=podcast_folder,
                transcription_path=transcription_path,
                audio_info_path=output_audio_info_file,
                audio_clips_folder=output_audio_clips_folder,
            )

            return "Processing complete", 200
        else:
            return "No key statements found", 200

    elif ext == ".mp4":
        # audio data
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        audio_file_path = os.path.join(podcast_folder, base_name + ".mp3")

        # run transcription and visual processing in parallel
        with ThreadPoolExecutor() as executor:
            future_transcription = executor.submit(process_transcription, file_path, podcast_folder)
            future_visual = executor.submit(process_visual, file_path, podcast_folder)

            (key_statements, silent_sections, summarized_statements, kmeans), transcription_path = future_transcription.result()
            visual_content_highlights = future_visual.result()

        if key_statements:
            output_audio_info_file = save_audio_info(os.path.basename(audio_file_path), key_statements, summarized_statements)
            output_audio_info_file = save_data_to_folder(output_audio_info_file, podcast_folder)

            output_audio_clips_folder = extract_and_save_audio_clips(audio_file_path, key_statements)
            output_audio_clips_folder = save_data_to_folder(output_audio_clips_folder, podcast_folder)
        else:
            return "No key statements found", 200

        # visual content data
        if visual_content_highlights:
            output_visual_content_info_file = save_visual_content_info(os.path.basename(file_path), visual_content_highlights)
            output_visual_content_info_file = save_data_to_folder(output_visual_content_info_file, podcast_folder)

            output_visual_content_folder = extract_and_save_visual_content_clips(file_path, visual_content_highlights)
            output_visual_content_folder = save_data_to_folder(output_visual_content_folder, podcast_folder)
        else:
            # create empty visual content file so the merge function can still work
            output_visual_content_info_file = os.path.join(podcast_folder, "visualContentInfo.json")
            with open(output_visual_content_info_file, "w", encoding="utf-8") as f:
                json.dump({"clips": []}, f, ensure_ascii=False, indent=4)
            output_visual_content_folder = None

        # merge data
        output_video_info_file, output_video_clips_folder = (merge_highlights_for_video_podcast(output_audio_info_file, output_visual_content_info_file, file_path, podcast_folder))

        # save podcast data
        save_podcast_data(
            email=email,
            filename=file.filename,
            podcast_folder=podcast_folder,
            transcription_path=transcription_path,
            audio_info_path=output_audio_info_file,
            audio_clips_folder=output_audio_clips_folder,
            visual_info_path=output_visual_content_info_file,
            visual_clips_folder=output_visual_content_folder,
            video_info_path=output_video_info_file,
            video_clips_folder=output_video_clips_folder,
        )

        return "Processing complete", 200

    else:
        return "Unsupported file type", 400


@app.route("/dashboardEnglish")
def dashboard_page_english():
    import string

    email = session.get("email")
    if not email:
        return redirect(url_for("login_english"))

    audio_data = load_audio_info()
    video_data = load_video_info()

    audio_podcasts = {pod["podcast_dir"]: pod for pod in audio_data.get("podcasts", [])}
    video_podcasts = {pod["podcast_dir"]: pod for pod in video_data.get("podcasts", [])}

    all_podcast_dirs = set(audio_podcasts) | set(video_podcasts)

    clips_by_podcast = {}

    for podcast_dir in all_podcast_dirs:
        podcast_name = (
            video_podcasts.get(podcast_dir, {}).get("podcast_name")
            or audio_podcasts.get(podcast_dir, {}).get("podcast_name")
            or podcast_dir
        )

        video_clips_path = os.path.join(data_folder, email, podcast_dir, "videoClips")
        audio_clips_path = os.path.join(data_folder, email, podcast_dir, "audioClips")

        video_clips = []
        audio_clips = []

        # collect video clips
        if os.path.exists(video_clips_path):
            mp4_files = sorted(
                [f for f in os.listdir(video_clips_path) if f.endswith(".mp4")],
                key=natural_sort_key,
            )
            if mp4_files:
                clips_info = video_podcasts.get(podcast_dir, {}).get("clips", [])
                for i, filename in enumerate(mp4_files):
                    clip_info = next((c for c in clips_info if c.get("clip_number") == i + 1), {})
                    summary = clip_info.get("summary_text", "No summary available")
                    cleaned_summary = summary.translate(str.maketrans("", "", string.punctuation))
                    video_clips.append(
                        {
                            "clip": f"{podcast_dir}/videoClips/{filename}",
                            "summary": cleaned_summary,
                            "type": "video",
                        }
                    )

        # if no video clips so collect audio clips
        if not video_clips and os.path.exists(audio_clips_path):
            audio_files = sorted(
                [
                    f
                    for f in os.listdir(audio_clips_path)
                    if f.endswith(".mp3") or f.endswith(".wav")
                ],
                key=natural_sort_key,
            )
            if audio_files:
                clips_info = audio_podcasts.get(podcast_dir, {}).get("clips", [])
                for i, filename in enumerate(audio_files):
                    clip_info = next((c for c in clips_info if c.get("clip_number") == i + 1), {})
                    summary = clip_info.get("summary_text", "No summary available")
                    cleaned_summary = summary.translate(str.maketrans("", "", string.punctuation))
                    audio_clips.append(
                        {
                            "clip": f"{podcast_dir}/audioClips/{filename}",
                            "summary": cleaned_summary,
                            "type": "audio",
                        }
                    )

        final_clips = video_clips if video_clips else audio_clips
        if final_clips:
            clips_by_podcast[podcast_name] = final_clips

    return render_custom_template("dashboardEnglish.html", clips_by_podcast=clips_by_podcast)


@app.route("/uploadArabic", methods=["GET", "POST"])
def upload_page_arabic():
    if request.method == "GET":
        return render_custom_template("uploadArabic.html", lang="arabic")

    if "podcast-file-input" not in request.files:
        return "No file part", 400

    email = session.get("email")
    if not email:
        return redirect(url_for("login_arabic"))

    file = request.files["podcast-file-input"]
    if file.filename == "":
        return "No selected file", 400

    # create user folder
    user_folder = os.path.join(data_folder, email)
    os.makedirs(user_folder, exist_ok=True)

    # save temporarily to compute checksum
    temp_path = os.path.join(user_folder, file.filename)
    file.save(temp_path)

    # check if podcast is Arabic language
    if not is_arabic_language(temp_path):
        os.remove(temp_path)
        return "Arabic podcasts only", 400

    checksum = get_file_checksum(temp_path)
    if is_duplicate_podcast(checksum, user_folder):
        os.remove(temp_path)
        return "Already been uploaded", 400

    # create new podcast folder
    podcast_folder = next_podcast_folder(user_folder)
    os.makedirs(podcast_folder)

    # move uploaded file into podcast folder
    file_path = os.path.join(podcast_folder, file.filename)
    shutil.move(temp_path, file_path)

    # save checksum
    save_checksum(checksum, podcast_folder)

    # determine file extension
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    if ext == ".mp3":
        # audio data
        transcription_path = process_audio_file(file_path)
        if not transcription_path.startswith(podcast_folder):
            transcription_path = save_data_to_folder(transcription_path, podcast_folder)

        key_statements, silent_sections, summarized_statements, kmeans = (summarize_audio_podcast(file_path, transcription_path))

        if key_statements:
            output_audio_info_file = save_audio_info(os.path.basename(file_path), key_statements, summarized_statements)
            output_audio_info_file = save_data_to_folder(output_audio_info_file, podcast_folder)

            output_audio_clips_folder = extract_and_save_audio_clips(file_path, key_statements)
            output_audio_clips_folder = save_data_to_folder(output_audio_clips_folder, podcast_folder)

            # save podcast data
            save_podcast_data(
                email=email,
                filename=file.filename,
                podcast_folder=podcast_folder,
                transcription_path=transcription_path,
                audio_info_path=output_audio_info_file,
                audio_clips_folder=output_audio_clips_folder,
            )

            return "Processing complete", 200
        else:
            return "No key statements found", 200

    elif ext == ".mp4":
        # audio data
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        audio_file_path = os.path.join(podcast_folder, base_name + ".mp3")

        # run transcription and visual processing in parallel
        with ThreadPoolExecutor() as executor:
            future_transcription = executor.submit(process_transcription, file_path, podcast_folder)
            future_visual = executor.submit(process_visual, file_path, podcast_folder)

            (key_statements, silent_sections, summarized_statements, kmeans), transcription_path = future_transcription.result()
            visual_content_highlights = future_visual.result()

        if key_statements:
            output_audio_info_file = save_audio_info(os.path.basename(audio_file_path), key_statements, summarized_statements)
            output_audio_info_file = save_data_to_folder(output_audio_info_file, podcast_folder)

            output_audio_clips_folder = extract_and_save_audio_clips(audio_file_path, key_statements)
            output_audio_clips_folder = save_data_to_folder(output_audio_clips_folder, podcast_folder)
        else:
            return "No key statements found", 200

        # visual content data
        if visual_content_highlights:
            output_visual_content_info_file = save_visual_content_info(os.path.basename(file_path), visual_content_highlights)
            output_visual_content_info_file = save_data_to_folder(output_visual_content_info_file, podcast_folder)

            output_visual_content_folder = extract_and_save_visual_content_clips(file_path, visual_content_highlights)
            output_visual_content_folder = save_data_to_folder(output_visual_content_folder, podcast_folder)
        else:
            # create empty visual content file so the merge function can still work
            output_visual_content_info_file = os.path.join(podcast_folder, "visualContentInfo.json")
            with open(output_visual_content_info_file, "w", encoding="utf-8") as f:
                json.dump({"clips": []}, f, ensure_ascii=False, indent=4)
            output_visual_content_folder = None

        # merge data
        output_video_info_file, output_video_clips_folder = (merge_highlights_for_video_podcast(output_audio_info_file, output_visual_content_info_file, file_path, podcast_folder))

        # save podcast data
        save_podcast_data(
            email=email,
            filename=file.filename,
            podcast_folder=podcast_folder,
            transcription_path=transcription_path,
            audio_info_path=output_audio_info_file,
            audio_clips_folder=output_audio_clips_folder,
            visual_info_path=output_visual_content_info_file,
            visual_clips_folder=output_visual_content_folder,
            video_info_path=output_video_info_file,
            video_clips_folder=output_video_clips_folder,
        )

        return "Processing complete", 200

    else:
        return "Unsupported file type", 400


@app.route("/dashboardArabic")
def dashboard_page_arabic():
    import string

    email = session.get("email")
    if not email:
        return redirect(url_for("login_arabic"))

    audio_data = load_audio_info()
    video_data = load_video_info()

    audio_podcasts = {pod["podcast_dir"]: pod for pod in audio_data.get("podcasts", [])}
    video_podcasts = {pod["podcast_dir"]: pod for pod in video_data.get("podcasts", [])}

    all_podcast_dirs = set(audio_podcasts) | set(video_podcasts)

    clips_by_podcast = {}

    for podcast_dir in all_podcast_dirs:
        podcast_name = (
            video_podcasts.get(podcast_dir, {}).get("podcast_name")
            or audio_podcasts.get(podcast_dir, {}).get("podcast_name")
            or podcast_dir
        )

        video_clips_path = os.path.join(data_folder, email, podcast_dir, "videoClips")
        audio_clips_path = os.path.join(data_folder, email, podcast_dir, "audioClips")

        video_clips = []
        audio_clips = []

        # collect video clips
        if os.path.exists(video_clips_path):
            mp4_files = sorted(
                [f for f in os.listdir(video_clips_path) if f.endswith(".mp4")],
                key=natural_sort_key,
            )
            if mp4_files:
                clips_info = video_podcasts.get(podcast_dir, {}).get("clips", [])
                for i, filename in enumerate(mp4_files):
                    clip_info = next((c for c in clips_info if c.get("clip_number") == i + 1), {})
                    summary = clip_info.get("summary_text", "No summary available")
                    cleaned_summary = summary.translate(str.maketrans("", "", string.punctuation))
                    video_clips.append(
                        {
                            "clip": f"{podcast_dir}/videoClips/{filename}",
                            "summary": cleaned_summary,
                            "type": "video",
                        }
                    )

        # if no video clips so collect audio clips
        if not video_clips and os.path.exists(audio_clips_path):
            audio_files = sorted(
                [
                    f
                    for f in os.listdir(audio_clips_path)
                    if f.endswith(".mp3") or f.endswith(".wav")
                ],
                key=natural_sort_key,
            )
            if audio_files:
                clips_info = audio_podcasts.get(podcast_dir, {}).get("clips", [])
                for i, filename in enumerate(audio_files):
                    clip_info = next((c for c in clips_info if c.get("clip_number") == i + 1), {})
                    summary = clip_info.get("summary_text", "No summary available")
                    cleaned_summary = summary.translate(str.maketrans("", "", string.punctuation))
                    audio_clips.append(
                        {
                            "clip": f"{podcast_dir}/audioClips/{filename}",
                            "summary": cleaned_summary,
                            "type": "audio",
                        }
                    )

        final_clips = video_clips if video_clips else audio_clips
        if final_clips:
            clips_by_podcast[podcast_name] = final_clips

    return render_custom_template("dashboardArabic.html", lang="arabic", clips_by_podcast=clips_by_podcast)


# helper route

@app.route("/audioClips/<podcast_name>/<filename>")
def serve_audio_clips(podcast_name, filename):
    email = session.get("email")
    if not email:
        abort(401)

    audio_clips_folder = os.path.join(os.getcwd(), data_folder, email, podcast_name, "audioClips")
    clip_path = os.path.join(audio_clips_folder, filename)
    if not os.path.exists(clip_path):
        abort(404)

    return send_from_directory(audio_clips_folder, filename)


@app.route("/videoClips/<podcast_name>/<filename>")
def serve_video_clips(podcast_name, filename):
    email = session.get("email")
    if not email:
        abort(401)

    video_clips_folder = os.path.join(os.getcwd(), data_folder, email, podcast_name, "videoClips")
    clip_path = os.path.join(video_clips_folder, filename)
    if not os.path.exists(clip_path):
        abort(404)

    return send_from_directory(video_clips_folder, filename)


@app.route("/deleteClip/<podcastDir>/<filename>", methods=["DELETE"])
def delete_clip(podcastDir, filename):
    email = session.get("email")
    if not email:
        return redirect(url_for("login"))

    user_folder = os.path.join(data_folder, email, podcastDir)

    # determine folder based on file type
    if filename.endswith(".mp4"):
        clip_folder = "videoClips"
        info_file = "videoInfo.json"
    else:
        clip_folder = "audioClips"
        info_file = "audioInfo.json"

    clip_path = os.path.join(user_folder, clip_folder, filename)
    info_path = os.path.join(user_folder, info_file)

    # delete the clip file if it exists
    if os.path.exists(clip_path):
        os.remove(clip_path)

    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)

        if "clips" in info_data:
            info_data["clips"] = [
                item for item in info_data["clips"] if item.get("clip_name") != filename
            ]
        elif "podcasts" in info_data:
            for podcast in info_data["podcasts"]:
                if podcast.get("podcast_dir") == podcastDir and "clips" in podcast:
                    podcast["clips"] = [
                        item
                        for item in podcast["clips"]
                        if item.get("clip_name") != filename
                    ]

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info_data, f, indent=4, ensure_ascii=False)

    return Response(status=204)



# DB


app.secret_key = "123321"  # needed for session
bcrypt = Bcrypt(app)


client = MongoClient("mongodb://localhost:27017/")
db = client["sawtDB"]
users_collection = db["users"]
podcasts_collection = db["podcasts"]


@app.route("/registrationEnglish", methods=["GET", "POST"])
def register_english():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        existing_user = users_collection.find_one({"email": email})

        if not existing_user:
            # only insert if its a new email
            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
            users_collection.insert_one({"name": name, "email": email, "password": hashed_pw})

        # in both cases the new or existing go to login
        return redirect(url_for("login_english"))

    return render_custom_template("registrationEnglish.html")


@app.route("/loginEnglish", methods=["GET", "POST"])
def login_english():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})

        if user and bcrypt.check_password_hash(user["password"], password):
            session["email"] = email
            return redirect(url_for("upload_page_english"))
        else:
            return render_custom_template("loginEnglish.html")

    return render_custom_template("loginEnglish.html")


@app.route("/settingsEnglish", methods=["GET", "POST"])
def settings_english():
    email = session.get("email")
    if not email:
        return redirect(url_for("login_english"))

    user = users_collection.find_one({"email": email})

    if request.method == "POST":
        name = request.form.get("name")
        new_email = request.form.get("email")
        password = request.form.get("password")

        update_data = {"name": name, "email": new_email}

        if password:
            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
            update_data["password"] = hashed_pw

        users_collection.update_one({"email": email}, {"$set": update_data})
        session["email"] = new_email

        return jsonify({"success": True})

    return render_custom_template("settingsEnglish.html", user=user)


@app.route("/registrationArabic", methods=["GET", "POST"])
def register_arabic():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        existing_user = users_collection.find_one({"email": email})

        if not existing_user:
            # only insert if its a new email
            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
            users_collection.insert_one({"name": name, "email": email, "password": hashed_pw})

        # in both cases the new or existing go to login
        return redirect(url_for("login_arabic"))

    return render_custom_template("registrationArabic.html", lang="arabic")


@app.route("/loginArabic", methods=["GET", "POST"])
def login_arabic():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})

        if user and bcrypt.check_password_hash(user["password"], password):
            session["email"] = email
            return redirect(url_for("upload_page_arabic"))
        else:
            return render_custom_template("loginArabic.html", lang="arabic")

    return render_custom_template("loginArabic.html", lang="arabic")


@app.route("/settingsArabic", methods=["GET", "POST"])
def settings_arabic():
    email = session.get("email")
    if not email:
        return redirect(url_for("login_arabic"))

    user = users_collection.find_one({"email": email})

    if request.method == "POST":
        name = request.form.get("name")
        new_email = request.form.get("email")
        password = request.form.get("password")

        update_data = {"name": name, "email": new_email}

        if password:
            hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
            update_data["password"] = hashed_pw

        users_collection.update_one({"email": email}, {"$set": update_data})
        session["email"] = new_email

        return jsonify({"success": True})
    
    return render_custom_template("settingsArabic.html", lang="arabic",user=user)


if __name__ == "__main__":
    app.run(debug=True)
