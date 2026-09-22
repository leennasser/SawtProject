import json
import os
import time
import cv2 as cv
import ffmpeg
import mediapipe as mp
import numpy as np
from collections import defaultdict
from keras_facenet import FaceNet
from numpy import dot
from numpy.linalg import norm
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceDetectorOptions


def cosine_similarity(vector_a, vector_b):
    return dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))


def calculate_total_time(timestamps, max_gap=3000):
    if not timestamps:
        return 0.0
    timestamps.sort()
    total_time = 0.0
    start = timestamps[0]
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > max_gap:
            total_time += timestamps[i - 1] - start
            start = timestamps[i]
    total_time += timestamps[-1] - start
    return total_time / 1000  # convert ms to seconds


def setup_face_detector(model_path, min_confidence=0.58):
    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=min_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


def extract_face_crops(frame, detections):
    faces = []
    h, w, _ = frame.shape
    for detection in detections:
        bbox = detection.bounding_box
        x1, y1 = max(0, int(bbox.origin_x)), max(0, int(bbox.origin_y))
        x2 = min(w, int(bbox.origin_x + bbox.width))
        y2 = min(h, int(bbox.origin_y + bbox.height))
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            face_resized = cv.resize(face, (160, 160))
            faces.append(face_resized)
    return faces


def process_video(
    video_path,
    model_path,
    frame_interval=20,
    similarity_threshold=0.7,
    false_positive_threshold=0.95,
):
    face_embedder = FaceNet()
    face_detector = setup_face_detector(model_path)

    face_appearances = defaultdict(list)
    known_faces = []
    total_faces_detected = 0

    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening file")
        return

    total_frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    current_frame_index = 0
    processing_start_time = time.time()
    person_id_counter = 0

    while current_frame_index < total_frame_count:
        video.set(cv.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(video.get(cv.CAP_PROP_POS_MSEC))

        detection_result = face_detector.detect_for_video(mediapipe_image, timestamp_ms)

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                x2 = int(bbox.origin_x + bbox.width)
                y2 = int(bbox.origin_y + bbox.height)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_images = extract_face_crops(frame, detection_result.detections)
            face_embeddings = face_embedder.embeddings(face_images)
            total_faces_detected += len(face_embeddings)

            for embedding in face_embeddings:
                matched_existing_person = False

                for label, known_embedding in known_faces:
                    similarity = cosine_similarity(embedding, known_embedding)

                    if similarity >= false_positive_threshold:
                        matched_existing_person = True
                        break

                    if similarity > similarity_threshold:
                        face_appearances[label].append(timestamp_ms)
                        print(f"Frame: {current_frame_index}, similarity: {similarity:.2f} -> {label}")
                        matched_existing_person = True
                        break

                if not matched_existing_person:
                    person_id_counter += 1
                    new_label = f"person{person_id_counter}"
                    known_faces.append((new_label, embedding))
                    face_appearances[new_label].append(timestamp_ms)
                    print(f"Frame: {current_frame_index}, new person detected -> {new_label}")

        cv.imshow("Face detection", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        current_frame_index += frame_interval

    video.release()
    cv.destroyAllWindows()

    print("Processing complete")
    total_processing_time = time.time() - processing_start_time
    return face_appearances, total_faces_detected, total_processing_time


def group_timestamps(timestamps, max_gap=3000):
    segments = []
    timestamps.sort()
    if not timestamps:
        return segments
    start = timestamps[0]
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > max_gap:
            segments.append((start, timestamps[i - 1]))
            start = timestamps[i]
    segments.append((start, timestamps[-1]))
    return segments


def merge_close_segments(segments, max_gap=3.0):
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["start_time"] - last["end_time"] <= max_gap:
            last["end_time"] = max(last["end_time"], seg["end_time"])
            last["duration"] = last["end_time"] - last["start_time"]
        else:
            merged.append(seg.copy())
    return merged


def remove_short_segments(segments, min_duration=3.0):
    return [s for s in segments if s["duration"] >= min_duration]


def assign_roles(face_appearances):
    all_segments = []
    total_times = {
        label: calculate_total_time(times) for label, times in face_appearances.items()
    }
    total_times = {k: v for k, v in total_times.items() if v >= 240}
    if not total_times:
        return []

    host_label = min(total_times, key=total_times.get)
    for label, times in face_appearances.items():
        if label not in total_times:
            continue
        role = "host" if label == host_label else "guest"
        for start, end in group_timestamps(times):
            all_segments.append(
                {
                    "role": role,
                    "start_time": start,
                    "end_time": end,
                    "duration": (end - start) / 1000,
                }
            )
    return all_segments


def detect_transition_highlights(face_appearances, window=6):
    segments = assign_roles(face_appearances)
    host_segments = [
        (s["start_time"] / 1000, s["end_time"] / 1000)
        for s in segments
        if s["role"] == "host"
    ]
    guest_segments = [
        (s["start_time"] / 1000, s["end_time"] / 1000)
        for s in segments
        if s["role"] == "guest"
    ]
    highlights = []

    # host → guest transitions
    for h_start, h_end in host_segments:
        for g_start, g_end in guest_segments:
            if 0 <= (g_start - h_end) <= window:
                highlights.append(
                    {
                        "role": "guest",
                        "start_time": h_start,
                        "end_time": g_end,
                        "duration": g_end - h_start,
                    }
                )
                break

    # guest → host transitions
    for g_start, g_end in guest_segments:
        for h_start, h_end in host_segments:
            if 0 <= (h_start - g_end) <= window:
                highlights.append(
                    {
                        "role": "host",
                        "start_time": g_start,
                        "end_time": h_end,
                        "duration": h_end - g_start,
                    }
                )
                break

    return remove_short_segments(merge_close_segments(highlights))


def detect_zscore_highlights(face_appearances, threshold=1.35):
    segments = assign_roles(face_appearances)
    role_durations = {"host": [], "guest": []}
    for seg in segments:
        role_durations[seg["role"]].append(seg["duration"])

    highlights = []
    for role in role_durations:
        dur = role_durations[role]
        if len(dur) < 2:
            continue
        mean, std = np.mean(dur), np.std(dur)
        for seg in segments:
            if seg["role"] == role and seg["duration"] > 25:
                z = (seg["duration"] - mean) / std
                if z >= threshold:
                    highlights.append(
                        {
                            "role": role,
                            "start_time": seg["start_time"] / 1000,
                            "end_time": seg["end_time"] / 1000,
                            "duration": seg["duration"],
                            "z_score": z,
                        }
                    )
    return highlights


def find_intersecting_highlights(highlights_a, highlights_b):
    final = []
    for a in highlights_a:
        for b in highlights_b:
            if a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]:
                start = max(a["start_time"], b["start_time"])
                end = min(a["end_time"], b["end_time"])
                if end - start > 25:
                    final.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "duration": end - start,
                            "role": a.get("role", "unknown"),
                        }
                    )
    return final


def save_visual_content_info(podcast_filename, highlights, output_file="visualContentInfo.json"):
    clips = [
        {
            "clip_number": i + 1,
            "start_time": h["start_time"],
            "end_time": h["end_time"],
            "duration": h["duration"],
        }
        for i, h in enumerate(highlights)
        if h["duration"] >= 25
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"podcast_name": podcast_filename, "clips": clips}, f, indent=4)

    print(f"Visual content information saved to file {output_file}")
    return output_file


def extract_and_save_visual_content_clips(video_path, highlights, output_folder="visualContentClips"):
    os.makedirs(output_folder, exist_ok=True)

    ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"The ffmpeg not found at: {ffmpeg_path}")

    clip_index = 1
    for clip in highlights:
        # ignore clip less than 25 second 
        if clip["duration"] < 25:
            continue

        output_path = os.path.join(output_folder, f"Clip{clip_index}.mp4")
        (
            ffmpeg.input(video_path, ss=clip["start_time"], t=clip["duration"])
            .output(output_path, c="copy")
            .run(
                cmd=ffmpeg_path, overwrite_output=True
            )  # this line forces it to use your local ffmpeg
        )
        clip_index += 1

    print(f"Clips saved to folder {output_folder}")
    return output_folder


def summarize_visual_content_podcast(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "blazeFaceShortRange.tflite")

    appearances, total_faces_detected, total_processing_time = process_video(file_path, model_path)

    highlights_a = detect_zscore_highlights(appearances, threshold=1.35)
    highlights_b = detect_transition_highlights(appearances)
    highlights_final = find_intersecting_highlights(highlights_a, highlights_b)

    return highlights_final


