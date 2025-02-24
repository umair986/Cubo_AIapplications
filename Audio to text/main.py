from fastapi import FastAPI, UploadFile, File
import whisper
import torch
import os
import json
import datetime
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# Initialize FastAPI app
app = FastAPI()

# Serve static files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")

# Load Whisper model (use GPU if available)
model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_srt_time(seconds):
    """Convert seconds to SRT timestamp format"""
    ms = int((seconds % 1) * 1000)
    time = str(datetime.timedelta(seconds=int(seconds)))
    return f"{time},{ms:03d}"

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    API Endpoint to receive an audio file, transcribe it, and return .txt, .srt, and .json files.
    """
    file_location = f"{OUTPUT_DIR}/{file.filename}"
    
    # Save uploaded file
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Transcribe audio
    response = model.transcribe(file_location)
    
    # Generate file paths
    base_name = os.path.splitext(file.filename)[0]
    txt_path = f"{OUTPUT_DIR}/{base_name}.txt"
    srt_path = f"{OUTPUT_DIR}/{base_name}.srt"
    json_path = f"{OUTPUT_DIR}/{base_name}.json"

    # Save .txt file (text only)
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(response["text"])

    # Save .json file (full response)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(response, json_file, ensure_ascii=False, indent=4)

    # Save .srt file (with timestamps)
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for segment in response["segments"]:
            start = format_srt_time(segment["start"])
            end = format_srt_time(segment["end"])
            text = segment["text"]
            srt_file.write(f"{start} --> {end}\n{text}\n\n")

    return {
        "message": "Transcription completed",
        "text_file": f"/download/{base_name}.txt",
        "srt_file": f"/download/{base_name}.srt",
        "json_file": f"/download/{base_name}.json"
    }

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """
    API endpoint to download the generated files.
    """
    file_path = f"{OUTPUT_DIR}/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    return {"error": "File not found"}
