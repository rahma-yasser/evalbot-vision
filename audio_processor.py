import json
import numpy as np
import librosa
import speech_recognition as sr
import scipy.signal
from google.cloud import storage

GCS_BUCKET_NAME = "interview-analysis-bucket"
storage_client = storage.Client()
gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

def suppress_noise(audio, sr=16000):
    audio = np.array(audio, dtype=np.float32)
    window = scipy.signal.windows.hann(512)
    stft = librosa.stft(audio, n_fft=512, hop_length=256, window=window)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    noise_mean = np.mean(magnitude[:, :10], axis=1)
    magnitude_clean = np.maximum(magnitude - noise_mean[:, np.newaxis], 0)
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=256, window=window)
    return audio_clean

def extract_prosodic_features(audio, sr):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[pitches > 0]
    intensity = librosa.feature.rms(y=audio)[0]
    features = {
        "duration": len(audio) / sr,
        "mean_pitch": np.mean(pitch_values) if len(pitch_values) > 0 else 0.0,
        "pitch_sd": np.std(pitch_values) if len(pitch_values) > 0 else 0.0,
        "intensityMean": np.mean(intensity) if len(intensity) > 0 else 0.0,
    }
    return features

def analyze_voice(prosodic_features):
    tension_score = prosodic_features["pitch_sd"] / 1000
    confidence_score = prosodic_features["intensityMean"] * 100 - prosodic_features["pitch_sd"] / 1000
    return tension_score, confidence_score

def process_audio(audio_path, question_number):
    audio_data, sr = librosa.load(audio_path, sr=16000)
    audio_data = suppress_noise(audio_data, sr)
    prosodic_features = extract_prosodic_features(audio_data, sr)
    tension, confidence = analyze_voice(prosodic_features)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            text = ""
    segment = {
        "question_number": question_number,
        "start_time": 0.0,
        "end_time": len(audio_data) / sr,
        "text": text,
        "prosodic_features": prosodic_features,
        "tension_score": tension,
        "confidence_score": confidence
    }
    return segment

def process_audio_background(audio_path, execution_id, question_number):
    segment = process_audio(audio_path, question_number)
    output_path = f"/tmp/audio_results_{question_number}.json"
    with open(output_path, "w") as f:
        json.dump([segment], f)
    gcs_output_key = f"audio_results/{execution_id}/{question_number}.json"
    blob = gcs_bucket.blob(gcs_output_key)
    blob.upload_from_filename(output_path)