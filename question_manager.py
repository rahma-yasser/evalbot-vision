import json
from google.cloud import storage

GCS_BUCKET_NAME = "interview-analysis-bucket"
storage_client = storage.Client()
gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

def extract_questions(data):
    if "questions" in data and isinstance(data["questions"], list):
        return data["questions"]
    elif "topics" in data and isinstance(data["topics"], list):
        questions = []
        for topic in data["topics"]:
            if "questions" in topic and isinstance(topic["questions"], list):
                questions.extend(topic["questions"])
        return questions
    return []

def get_question(execution_id, question_number, bucket_name="interview-analysis-bucket"):
    s3_key = f"{execution_id}/questions.json"
    try:
        blob = gcs_bucket.blob(s3_key)
        json_content = blob.download_as_string().decode("utf-8")
        data = json.loads(json_content)
        questions = extract_questions(data)
        if 1 <= question_number <= len(questions):
            return questions[question_number - 1]["question"]
        return None
    except Exception:
        return None