import re
import json
import torch
import numpy as np
import librosa
import os
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline


# detect silent sections in audio file
def detect_silence(file_path, silence_threshold=0.01):
    try:
        audio_signal, sample_rate = librosa.load(file_path, sr=None)
    except Exception as e:
        return []  # return empty list if there is an error

    non_silent_intervals = librosa.effects.split(audio_signal, top_db=silence_threshold)
    silent_sections = []
    prev_end = 0
    for start_sample, end_sample in non_silent_intervals:
        if start_sample > prev_end:
            silent_sections.append((prev_end / sample_rate, start_sample / sample_rate))
        prev_end = end_sample

    return silent_sections


# clean text file
def extract_clean_text(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    words, timestamps = [], []
    arabic_punctuation_pattern = r"[^\u0600-\u06FF\s،.؟!؛]"

    for entry in data:
        word = re.sub(arabic_punctuation_pattern, "", entry["word"])
        if word.strip():
            words.append(word)
            timestamps.append((entry["start_time"], entry["end_time"]))

    return words, timestamps


# dynamically split text into chunks while maintaining sentence coherence
def chunk_text(words, timestamps, max_chunk_size=300):
    chunks, chunk_times = [], []
    current_chunk, current_timestamps = [], []
    current_length = 0

    for i, word in enumerate(words):
        word_length = len(word)
        if current_length + word_length + len(current_chunk) > max_chunk_size:
            if current_chunk[-1][-1] in [".", "؟", "!"]:
                chunks.append(" ".join(current_chunk))
                chunk_times.append((current_timestamps[0][0], current_timestamps[-1][1]))
                current_chunk, current_timestamps = [word], [timestamps[i]]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_timestamps.append(timestamps[i])
                current_length += word_length
        else:
            current_chunk.append(word)
            current_timestamps.append(timestamps[i])
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        chunk_times.append((current_timestamps[0][0], current_timestamps[-1][1]))

    return chunks, chunk_times


# load AraBERT model and tokenizer
arabert_model = None
arabert_tokenizer = None
def get_arabert_model():
    global arabert_model, arabert_tokenizer

    if arabert_model is None or arabert_tokenizer is None:
        try:
            arabert_model_name = "aubmindlab/bert-base-arabertv02"
            arabert_tokenizer = AutoTokenizer.from_pretrained(arabert_model_name)
            arabert_model = AutoModel.from_pretrained(arabert_model_name)
        except Exception as e:
            print(f"Error loading AraBERT model: {e}")
            arabert_model = None
            arabert_tokenizer = None

    return arabert_model, arabert_tokenizer


# generate embeddings for each text chunk using AraBERT
def get_text_embeddings(text_chunks):
    arabert_model, arabert_tokenizer = get_arabert_model()

    sentence_embeddings = []
    for chunk in text_chunks:
        inputs = arabert_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)

        # check length of tokenized input before passing to the model
        token_length = len(inputs["input_ids"][0])
        if token_length > 512:
            print(f"The token length of this chunk exceeds the limit")

        with torch.no_grad():
            output = arabert_model(**inputs)

        sentence_embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())

    return np.array(sentence_embeddings)


# compute similarity scores for each chunk based on its closeness to the centroid
def calculate_similarity_scores(embeddings):
    centroid = np.mean(embeddings, axis=0)
    return cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()


# determine the optimal number of clusters dynamically
def determine_cluster_count(num_chunks, base_clusters=3, max_clusters=10):
    scaling_factor = 0.1
    return min(int(base_clusters + scaling_factor * num_chunks), max_clusters)


# apply KMeans clustering on the text embeddings
def cluster_text(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans


# select the key statements dynamically
def select_key_statements(text_chunks, chunk_times, embeddings, silent_sections, num_clusters):
    max_statements = min(num_clusters * 2, len(text_chunks))

    scores = calculate_similarity_scores(embeddings)
    
    ranked_indices = np.argsort(scores)[::-1]  # sort in descending order
    key_statements = []
    seen_texts = set()  # to track already added statements and avoid duplicates

    # include statements appearing right after silence
    for silence_start, silence_end in silent_sections:
        for i, chunk_time in enumerate(chunk_times):
            chunk_start, chunk_end = chunk_time
            if chunk_start >= silence_end:
                statement_text = text_chunks[i]
                if statement_text not in seen_texts:
                    key_statements.append(
                        {
                            "text": statement_text,
                            "start_time": chunk_start,
                            "end_time": chunk_end,
                        }
                    )
                    seen_texts.add(statement_text)
                break  # move to the next silence section

    # fill up remaining slots with highest ranked statements
    remaining_slots = max_statements - len(key_statements)
    for idx in ranked_indices[:remaining_slots]:
        statement_text = text_chunks[idx]
        if statement_text not in seen_texts:
            key_statements.append(
                {
                    "text": statement_text,
                    "start_time": chunk_times[idx][0],
                    "end_time": chunk_times[idx][1],
                }
            )
            seen_texts.add(statement_text)

    return key_statements


# load summarization model for labeled clips
summarization_model = None
def get_summarization_model():
    global summarization_model

    if summarization_model is None:
        try:
            mt5_model_name = "csebuetnlp/mT5_multilingual_XLSum"
            summarization_model = pipeline("summarization", model=mt5_model_name, framework="pt")
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            summarization_model = None
    return summarization_model


# summary key statement for make label
def summarize_text_for_label(text):
    summarization_model = get_summarization_model()

    if not summarization_model:
        return text

    summary = summarization_model(text, max_length=60, min_length=15, do_sample=False)
    return summary[0]["summary_text"]


# extract and save audio clips based on key statements
def extract_and_save_audio_clips(file_path, key_statements, margin_seconds=0.5, output_audio_folder="audioClips"):
    audio_signal, sample_rate = librosa.load(file_path, sr=None)

    # ensure the output folder exists
    if not os.path.exists(output_audio_folder):
        os.makedirs(output_audio_folder)

    for idx, statement in enumerate(key_statements):
        start_sample = int(statement["start_time"] * sample_rate)
        end_sample = int((statement["end_time"] + margin_seconds) * sample_rate)
        end_sample = min(end_sample, len(audio_signal))

        clip = audio_signal[start_sample:end_sample]

        clip_path = os.path.join(output_audio_folder, f"Clip{idx + 1}.wav")
        sf.write(clip_path, clip, sample_rate)  # save as WAV format

    print(f"Clips saved to folder {output_audio_folder}")
    return output_audio_folder


# save the audio information in the file
def save_audio_info(podcast_filename, key_statements, summaries, output_audio_file="audioInfo.json"):
    clips = []

    for idx, statement in enumerate(key_statements):
        clip_info = {
            "clip_number": idx + 1,
            "key_text": statement["text"],
            "summary_text": summaries[idx],
            "start_time": statement["start_time"],
            "end_time": statement["end_time"],
        }
        clips.append(clip_info)

    data = {
        "podcast_name": podcast_filename,
        "clips": clips
    }

    with open(output_audio_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Audio information saved to file {output_audio_file}")
    return output_audio_file


# generate a summary of the audio podcast
def summarize_audio_podcast(file_path, transcript_file, max_chunk_size=300):
    silent_sections = detect_silence(file_path)
    words, timestamps = extract_clean_text(transcript_file)
    text_chunks, chunk_times = chunk_text(words, timestamps, max_chunk_size)

    if not text_chunks:
        return [], silent_sections, None  # return empty summary if no text is available

    num_clusters = determine_cluster_count(len(text_chunks))
    embeddings = get_text_embeddings(text_chunks)
    kmeans = cluster_text(embeddings, num_clusters)
    key_statements = select_key_statements(text_chunks, chunk_times, embeddings, silent_sections, num_clusters)
    summarized_statements = [summarize_text_for_label(statement["text"]) for statement in key_statements]

    return key_statements, silent_sections, summarized_statements, kmeans

