import os
import sys
import uvicorn
from google.cloud import storage

GCS_BUCKET_NAME = "interview-analysis-bucket"
storage_client = storage.Client()
gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Download scripts from GCS
def download_scripts():
    scripts = ["main.py", "audio_processor.py", "vision_processor.py", "question_manager.py"]
    os.makedirs("/tmp/scripts", exist_ok=True)
    for script in scripts:
        local_path = f"/tmp/scripts/{script}"
        try:
            blob = gcs_bucket.blob(f"scripts/{script}")
            blob.download_to_filename(local_path)
            print(f"Downloaded {script} to {local_path}")
        except Exception as e:
            print(f"Error downloading {script}: {e}")
            raise

# Download model from GCS
def download_model():
    model_path = "/models/best_model3.pth"
    if not os.path.exists(model_path):
        os.makedirs("/models", exist_ok=True)
        try:
            blob = gcs_bucket.blob("models/best_model3.pth")
            blob.download_to_filename(model_path)
            print(f"Downloaded model to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

# Download service account key from GCS
def download_service_account_key():
    key_path = "/app/key.json"
    if not os.path.exists(key_path):
        try:
            blob = gcs_bucket.blob("credentials/key.json")
            blob.download_to_filename(key_path)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
            print(f"Downloaded service account key to {key_path}")
        except Exception as e:
            print(f"Error downloading service account key: {e}")
            raise

if __name__ == "__main__":
    print("Starting bootstrap process...")
    try:
        download_service_account_key()
        download_scripts()
        download_model()
        print("All files downloaded successfully.")
        sys.path.append("/tmp/scripts")
        port = int(os.getenv("PORT", 8080))
        print(f"Starting FastAPI app on port {port}...")
        uvicorn.run("main:app", host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)