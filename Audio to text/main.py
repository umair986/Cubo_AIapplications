from fastapi import FastAPI, UploadFile, File
import whisper
import torch
import os

app = FastAPI()

# Load Whisper model (automatically uses CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Uploads an audio file and returns the transcription as a JSON response.
    """
    file_location = f"uploads/{file.filename}"
    
    # Ensure 'uploads' folder exists
    os.makedirs("uploads", exist_ok=True)

    # Save the uploaded file
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Transcribe the audio file
    result = model.transcribe(file_location)

    # Return Whisper's output as JSON
    return result
