import os
import sys
import importlib.util
from google.cloud import storage

GCS_BUCKET_NAME = "interview-analysis-bucket"
storage_client = storage.Client()
gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

def download_scripts():
    scripts = ["main.py", "audio_processor.py", "vision_processor.py", "question_manager.py"]
    os.makedirs("/tmp/scripts", exist_ok=True)
    for script in scripts:
        local_path = f"/tmp/scripts/{script}"
        blob = gcs_bucket.blob(f"scripts/{script}")
        blob.download_to_filename(local_path)
        print(f"Downloaded {script} to {local_path}")

def download_model():
    model_path = "/models/best_model3.pth"
    if not os.path.exists(model_path):
        os.makedirs("/models", exist_ok=True)
        blob = gcs_bucket.blob("models/best_model3.pth")
        blob.download_to_filename(model_path)
        print(f"Downloaded model to {model_path}")

def run_app():
    sys.path.append("/tmp/scripts")
    spec = importlib.util.spec_from_file_location("main", "/tmp/scripts/main.py")
    main_module = importlib.util.module_from_spec(spec)
    sys.modules["main"] = main_module
    spec.loader.exec_module(main_module)

if __name__ == "__main__":
    print("Downloading scripts and model...")
    download_scripts()
    download_model()
    print("Starting FastAPI app...")
    run_app()