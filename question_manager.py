from google.cloud import storage
import json

def get_question(execution_id, question_number, bucket_name):
    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(bucket_name)
    gcs_key = f"{execution_id}/questions.json"
    questions_path = f"/tmp/{execution_id}_questions.json"
    try:
        blob = gcs_bucket.blob(gcs_key)
        blob.download_to_filename(questions_path)
        with open(questions_path, "r") as f:
            questions = json.load(f)["questions"]
        return questions[question_number - 1]["question"]
    except Exception as e:
        print(f"Error getting question: {e}")
        return None