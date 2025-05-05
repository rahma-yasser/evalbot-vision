import json
import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage
from audio_processor import process_audio_background
from vision_processor import process_video_background, process_cheating_background, combine_results_background
from question_manager import get_question

app = FastAPI(title="Interview Analysis API")
GCS_BUCKET_NAME = "interview-analysis-bucket"
storage_client = storage.Client()
gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

os.makedirs("/tmp", exist_ok=True)

class InterviewRequest(BaseModel):
    user_skills: str = "General"
    job_title: str = "General"
    max_questions: int = 5

def generate_questions(user_skills, job_title, max_questions):
    questions = [
        {"question": "Tell me about yourself."},
        {"question": "What are your strengths?"},
        {"question": "Why do you want this job?"},
    ]
    return questions[:max_questions]

@app.post("/start_interview")
async def start_interview(request: InterviewRequest):
    execution_id = str(uuid.uuid4())
    questions = generate_questions(request.user_skills, request.job_title, request.max_questions)
    gcs_key = f"{execution_id}/questions.json"
    blob = gcs_bucket.blob(gcs_key)
    blob.upload_from_string(json.dumps({"questions": questions}))
    return JSONResponse({
        "message": "Interview started",
        "execution_id": execution_id,
        "first_question": questions[0]["question"]
    })

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...), execution_id: str = None, question_number: int = 1, background_tasks: BackgroundTasks = BackgroundTasks()):
    if not execution_id:
        raise HTTPException(status_code=400, detail="execution_id is required")
    
    audio_path = f"/tmp/audio_{question_number}.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    gcs_audio_key = f"inputs/audio/{execution_id}/{question_number}.wav"
    blob = gcs_bucket.blob(gcs_audio_key)
    blob.upload_from_filename(audio_path)
    
    background_tasks.add_task(process_audio_background, audio_path, execution_id, question_number)
    
    return JSONResponse({
        "message": "Audio uploaded and processing started",
        "execution_id": execution_id,
        "question_number": question_number
    })

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...), execution_id: str = None, background_tasks: BackgroundTasks = BackgroundTasks()):
    if not execution_id:
        raise HTTPException(status_code=400, detail="execution_id is required")
    
    video_path = f"/tmp/video.mp4"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    gcs_video_key = f"inputs/video/{execution_id}.mp4"
    blob = gcs_bucket.blob(gcs_video_key)
    blob.upload_from_filename(video_path)
    
    audio_results = []
    for i in range(1, 6):
        gcs_audio_result_key = f"audio_results/{execution_id}/{i}.json"
        audio_result_path = f"/tmp/audio_results_{i}.json"
        try:
            blob = gcs_bucket.blob(gcs_audio_result_key)
            blob.download_to_filename(audio_result_path)
            with open(audio_result_path, "r") as f:
                audio_results.extend(json.load(f))
        except:
            continue
    
    background_tasks.add_task(process_video_background, video_path, audio_results, execution_id)
    background_tasks.add_task(process_cheating_background, video_path, execution_id)
    background_tasks.add_task(combine_results_background, audio_results, execution_id)
    
    return JSONResponse({
        "message": "Video uploaded and processing started",
        "execution_id": execution_id
    })

@app.get("/get_question")
async def get_question_endpoint(execution_id: str, question_number: int):
    question = get_question(execution_id, question_number, "interview-analysis-bucket")
    if question:
        return JSONResponse({"question": question})
    raise HTTPException(status_code=404, detail="Question not found")

@app.get("/get_results")
async def get_results(execution_id: str):
    gcs_results_key = f"combined_results/{execution_id}.json"
    results_path = f"/tmp/combined_results.json"
    try:
        blob = gcs_bucket.blob(gcs_results_key)
        blob.download_to_filename(results_path)
        with open(results_path, "r") as f:
            results = json.load(f)
        return JSONResponse({"results": results, "execution_id": execution_id})
    except:
        raise HTTPException(status_code=404, detail="Results not found or processing not complete")

@app.get("/status")
async def get_status(execution_id: str):
    status = {
        "execution_id": execution_id,
        "audio_processed": [],
        "video_processed": False,
        "cheating_processed": False,
        "results_combined": False
    }
    for i in range(1, 6):
        gcs_audio_result_key = f"audio_results/{execution_id}/{i}.json"
        try:
            blob = gcs_bucket.blob(gcs_audio_result_key)
            if blob.exists():
                status["audio_processed"].append(i)
        except:
            continue
    try:
        if gcs_bucket.blob(f"vision_results/{execution_id}.json").exists():
            status["video_processed"] = True
        if gcs_bucket.blob(f"cheating_results/{execution_id}.json").exists():
            status["cheating_processed"] = True
        if gcs_bucket.blob(f"combined_results/{execution_id}.json").exists():
            status["results_combined"] = True
    except:
        pass
    return JSONResponse(status)