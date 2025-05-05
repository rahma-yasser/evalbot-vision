import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import mediapipe as mp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from google.cloud import storage

GCS_BUCKET_NAME = "interview-analysis-bucket"
storage_client = storage.Client()
gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

class CustomEfficientNetV2(nn.Module):
    def __init__(self):
        super(CustomEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(emotion_classes)),
        )

    def forward(self, x):
        return self.model(x)

transform = Compose([
    Resize(224, 224),
    Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2(),
])

def analyze_emotion(probabilities):
    max_prob, predicted_idx = torch.max(probabilities, dim=1)
    predicted_emotion = emotion_classes[predicted_idx.item()]
    confidence_score = max_prob.item() * 100 if predicted_emotion in {"Happy", "Neutral"} else 0.0
    tension_score = max_prob.item() * 100 if predicted_emotion in {"Angry", "Fear", "Sad", "Disgust"} else 0.0
    emotion_dist = {emotion: prob.item() for emotion, prob in zip(emotion_classes, probabilities[0])}
    return predicted_emotion, confidence_score, tension_score, emotion_dist

def process_video(video_path, audio_results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomEfficientNetV2().to(device)
    model.load_state_dict(torch.load("/models/best_model3.pth", map_location=device))
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = frame_number / frame_rate
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detection.process(rgb_frame)
        frame_data = {"timestamp": timestamp, "emotions": []}
        if detections.detections:
            for detection in detections.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face = rgb_frame[y:y+h, x:x+w]
                face = transform(image=face)["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(face)
                    probabilities = F.softmax(output, dim=1)
                    emotion, conf, tens, dist = analyze_emotion(probabilities)
                frame_data["emotions"].append({
                    "emotion": emotion,
                    "confidence_score": conf,
                    "tension_score": tens,
                    "emotion_distribution": dist,
                    "bbox": [x, y, w, h]
                })
        frame_results.append(frame_data)
    cap.release()

    question_results = []
    for question in audio_results:
        q_start, q_end, q_number = question["start_time"], question["end_time"], question["question_number"]
        relevant_frames = [f for f in frame_results if q_start <= f["timestamp"] <= q_end]
        conf_scores, tens_scores, emotion_dists = [], [], []
        for frame in relevant_frames:
            for emotion in frame["emotions"]:
                conf_scores.append(emotion["confidence_score"])
                tens_scores.append(emotion["tension_score"])
                emotion_dists.append(emotion["emotion_distribution"])
        question_results.append({
            "question_number": q_number,
            "start_time": q_start,
            "end_time": q_end,
            "average_confidence_score": np.mean(conf_scores) if conf_scores else 0.0,
            "average_tension_score": np.mean(tens_scores) if tens_scores else 0.0,
            "average_emotion_distribution": {e: np.mean([d[e] for d in emotion_dists]) for e in emotion_classes} if emotion_dists else {e: 0.0 for e in emotion_classes}
        })

    all_conf_scores = [e["confidence_score"] for f in frame_results for e in f["emotions"]]
    all_tens_scores = [e["tension_score"] for f in frame_results for e in f["emotions"]]
    all_emotion_dists = [e["emotion_distribution"] for f in frame_results for e in f["emotions"]]
    
    results = {
        "per_question": question_results,
        "overall": {
            "average_confidence_score": np.mean(all_conf_scores) if all_conf_scores else 0.0,
            "average_tension_score": np.mean(all_tens_scores) if all_tens_scores else 0.0,
            "average_emotion_distribution": {e: np.mean([d[e] for d in all_emotion_dists]) for e in emotion_classes} if all_emotion_dists else {e: 0.0 for e in emotion_classes}
        },
        "raw_frames": frame_results
    }
    return results

def process_video_background(video_path, audio_results, execution_id):
    results = process_video(video_path, audio_results)
    output_path = f"/tmp/vision_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f)
    gcs_output_key = f"vision_results/{execution_id}.json"
    blob = gcs_bucket.blob(gcs_output_key)
    blob.upload_from_filename(output_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_iris_center(iris_points, mesh_points):
    iris_array = np.array([mesh_points[p] for p in iris_points])
    (cx, cy), radius = cv2.minEnclosingCircle(iris_array)
    return np.array([int(cx), int(cy)]), radius

def detect_gaze(eye_points, iris_center, mesh_points):
    eye_array = np.array([mesh_points[p] for p in eye_points])
    min_x, max_x = np.min(eye_array[:, 0]), np.max(eye_array[:, 0])
    relative_x = (iris_center[0] - min_x) / (max_x - min_x + 1e-6)
    return "Left" if relative_x < 0.4 else "Right" if relative_x > 0.5 else "Center"

def detect_head_rotation(mesh_points):
    nose_tip = mesh_points[1]
    left_eye = np.mean([mesh_points[p] for p in LEFT_EYE], axis=0)
    right_eye = np.mean([mesh_points[p] for p in RIGHT_EYE], axis=0)
    eyes_midpoint = (left_eye + right_eye) / 2
    diff = nose_tip[0] / mesh_points[0][0] - eyes_midpoint[0] / mesh_points[0][0]
    return "Right" if diff > 0.03 else "Left" if diff < -0.03 else "Straight"

def process_cheating(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results, alerts = [], []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    warnings, look_away_start, terminated = 0, None, False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = frame_number / frame_rate
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        
        frame_data = {"timestamp": timestamp, "gaze_direction": "Center", "head_rotation": "Straight", "multiple_faces": False}
        if output.multi_face_landmarks:
            frame_data["multiple_faces"] = len(output.multi_face_landmarks) > 1
            landmarks = output.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            mesh_points = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks])
            left_iris, _ = get_iris_center(LEFT_IRIS, mesh_points)
            right_iris, _ = get_iris_center(RIGHT_IRIS, mesh_points)
            left_gaze = detect_gaze(LEFT_EYE, left_iris, mesh_points)
            right_gaze = detect_gaze(RIGHT_EYE, right_iris, mesh_points)
            frame_data["gaze_direction"] = left_gaze if left_gaze == right_gaze else "Center"
            frame_data["head_rotation"] = detect_head_rotation(mesh_points)

            if frame_data["multiple_faces"]:
                alerts.append({"timestamp": timestamp, "type": "Multiple faces detected"})
            if frame_data["head_rotation"] != "Straight":
                alerts.append({"timestamp": timestamp, "type": f"Head turned {frame_data['head_rotation']}"})
            if frame_data["gaze_direction"] != "Center":
                if not look_away_start:
                    look_away_start = timestamp
                elif timestamp - look_away_start >= 5.0:
                    if warnings < 2:
                        warnings += 1
                        alerts.append({"timestamp": timestamp, "type": f"Warning {warnings}: Please focus on the camera."})
                        look_away_start = None
                    else:
                        alerts.append({"timestamp": timestamp, "type": "Interview terminated due to repeated looking away"})
                        terminated = True
                        break
            else:
                look_away_start = None

        frame_results.append(frame_data)
    cap.release()
    return {"frames": frame_results, "alerts": alerts, "terminated": terminated}

def process_cheating_background(video_path, execution_id):
    results = process_cheating(video_path)
    output_path = f"/tmp/cheating_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f)
    gcs_output_key = f"cheating_results/{execution_id}.json"
    blob = gcs_bucket.blob(gcs_output_key)
    blob.upload_from_filename(output_path)

def combine_results(audio_results, vision_results, cheating_results):
    combined_results = {"interview": []}
    for audio_segment in audio_results:
        question_number = audio_segment["question_number"]
        vision_segment = next((q for q in vision_results["per_question"] if q["question_number"] == question_number), None)
        cheating_frames = [f for f in cheating_results["frames"] if audio_segment["start_time"] <= f["timestamp"] <= audio_segment["end_time"]]
        cheating_alerts = [a for a in cheating_results["alerts"] if audio_segment["start_time"] <= a["timestamp"] <= audio_segment["end_time"]]
        
        question_data = {
            "question_number": question_number,
            "start_time": audio_segment["start_time"],
            "end_time": audio_segment["end_time"],
            "text": audio_segment["text"],
            "audio_analysis": {
                "prosodic_features": audio_segment["prosodic_features"],
                "tension_score": audio_segment["tension_score"],
                "confidence_score": audio_segment["confidence_score"]
            },
            "vision_analysis": vision_segment or {"average_confidence_score": 0.0, "average_tension_score": 0.0, "average_emotion_distribution": {e: 0.0 for e in emotion_classes}},
            "cheating_analysis": {"alerts": cheating_alerts}
        }
        combined_results["interview"].append(question_data)
    
    combined_results["overall"] = vision_results["overall"]
    combined_results["overall"]["cheating_alerts"] = cheating_results["alerts"]
    return combined_results

def combine_results_background(audio_results, execution_id):
    vision_results_path = f"/tmp/vision_results.json"
    cheating_results_path = f"/tmp/cheating_results.json"
    blob = gcs_bucket.blob(f"vision_results/{execution_id}.json")
    blob.download_to_filename(vision_results_path)
    blob = gcs_bucket.blob(f"cheating_results/{execution_id}.json")
    blob.download_to_filename(cheating_results_path)
    
    with open(vision_results_path, "r") as f:
        vision_results = json.load(f)
    with open(cheating_results_path, "r") as f:
        cheating_results = json.load(f)
    
    combined_results = combine_results(audio_results, vision_results, cheating_results)
    output_path = f"/tmp/combined_results.json"
    with open(output_path, "w") as f:
        json.dump(combined_results, f)
    gcs_output_key = f"combined_results/{execution_id}.json"
    blob = gcs_bucket.blob(gcs_output_key)
    blob.upload_from_filename(output_path)