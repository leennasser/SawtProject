import os
import json
import subprocess


def time_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def extract_the_clips(ffmpeg_path, input_video, start_time, end_time, output_path):
    duration = end_time - start_time
    command = [
        ffmpeg_path,
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        return False


def merge_highlights_for_video_podcast(output_audio_info_file, output_visual_content_file, video_path, output_folder):
    # load audio and visual content information
    with open(output_audio_info_file, "r", encoding="utf-8") as f:
        audio_data = json.load(f)

    with open(output_visual_content_file, "r", encoding="utf-8") as f:
        visual_data = json.load(f)

    audio_clips = audio_data.get("clips", [])
    visual_clips = visual_data.get("clips", [])

    merged_clips = []
    clip_index = 1

    for a_clip in audio_clips:
        a_start = a_clip["start_time"]
        a_end = a_clip["end_time"]
        overlap_found = False

        for v_clip in visual_clips:
            v_start = v_clip["start_time"]
            v_end = v_clip["end_time"]

            if time_overlap(a_start, a_end, v_start, v_end):
                merged_start = max(a_start, v_start)
                merged_end = min(a_end, v_end)
                duration = merged_end - merged_start

                if duration >= 25:
                    merged_clips.append(
                        {
                            "clip_number": clip_index,
                            "start_time": merged_start,
                            "end_time": merged_end,
                            "key_text": a_clip["key_text"],
                            "summary_text": a_clip["summary_text"],
                        }
                    )
                    clip_index += 1
                    overlap_found = True
                    break 

        duration = a_end - a_start
        # ignore clip less than 25 second
        if not overlap_found and duration >= 25:
            merged_clips.append(
                {
                    "clip_number": clip_index,
                    "start_time": a_start,
                    "end_time": a_end,
                    "key_text": a_clip["key_text"],
                    "summary_text": a_clip["summary_text"],
                }
            )
            clip_index += 1

    # save merged information
    os.makedirs(output_folder, exist_ok=True)
    output_video_info_file = os.path.join(output_folder, "videoInfo.json")
    with open(output_video_info_file, "w", encoding="utf-8") as f:
        json.dump(
            {"podcast_name": os.path.basename(video_path), "clips": merged_clips},
            f,
            ensure_ascii=False,
            indent=4,
        )
    print("Video information saved to file", os.path.basename(output_video_info_file))

    # create the clips
    output_video_clips_folder = os.path.join(output_folder, "videoClips")
    os.makedirs(output_video_clips_folder, exist_ok=True)

    ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"The ffmpeg not found at: {ffmpeg_path}")

    for clip in merged_clips:
        out_path = os.path.join(output_video_clips_folder, f"Clip{clip['clip_number']}.mp4")
        success = extract_the_clips(ffmpeg_path, video_path, clip["start_time"], clip["end_time"], out_path)
    print("Clips saved to folder", os.path.basename(output_video_clips_folder))

    print("Merge information completed")
    return output_video_info_file, output_video_clips_folder
