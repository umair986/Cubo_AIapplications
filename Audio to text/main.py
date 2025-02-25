from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import whisper
import torch
import os

app = FastAPI()

# Define a fixed API token (You can change this)
API_TOKEN = "Cubo123!@#"

# Load Whisper model (automatically uses CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    token: str = Header(None)  # Token should be passed in headers
):
    """
    Uploads an audio file, verifies the token, transcribes it, and returns the result as JSON.
    """

    # Check if the token is provided and valid
    if token is None or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")

    file_location = f"uploads/{file.filename}"
    
    # Ensure 'uploads' folder exists
    os.makedirs("uploads", exist_ok=True)

    # Save the uploaded file
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Transcribe the audio file
    result = model.transcribe(file_location)

    # Extract detected language
    detected_language = result.get("language", "Unknown")
    
    # Print the detected language
    print(f"Detected Language: {detected_language}")

    # Return Whisper's full JSON response along with the detected language
    return {
        "text": result["text"],
        "segments": result["segments"],
        "language": detected_language
    }
