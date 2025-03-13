import os
import io
import re
import queue
import threading
import google.generativeai as genai
import google.cloud.texttospeech as tts
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
tts_client = tts.TextToSpeechClient()

# Enhanced voice configuration
VOICE_CONFIG = {
    "en-US": {
        "name": "en-US-Neural2-J",
        "speaking_rate": 1.1,
        "pitch": 2.0
    },
    "hi-IN": {
        "name": "hi-IN-Neural2-D",
        "speaking_rate": 1.0,
        "pitch": 1.5
    },
    "ar-XA": {
        "name": "ar-XA-Standard-D",
        "speaking_rate": 0.95,
        "pitch": 1.2
    }
}

# Thread-safe queues and flags
tts_queue = queue.Queue()
interrupt_flag = threading.Event()
current_playback = None

def detect_language(text):
    if re.search(r'[\u0900-\u097F]', text): return "hi-IN"
    if re.search(r'[\u0600-\u06FF]', text): return "ar-XA"
    return "en-US"

def stream_gemini_response(prompt):
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(
        [SYSTEM_PROMPT, prompt],
        stream=True
    )
    for chunk in response:
        yield chunk.text

def tts_worker():
    global current_playback
    while True:
        text_chunk, lang_code = tts_queue.get()
        if text_chunk is None: break
        
        config = VOICE_CONFIG.get(lang_code, VOICE_CONFIG["en-US"])
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3,
            speaking_rate=config["speaking_rate"],
            pitch=config["pitch"]
        )
        
        response = tts_client.synthesize_speech(
            input=tts.SynthesisInput(text=text_chunk),
            voice=tts.VoiceSelectionParams(
                language_code=lang_code,
                name=config["name"]
            ),
            audio_config=audio_config
        )
        
        if not interrupt_flag.is_set():
            audio = AudioSegment.from_file(io.BytesIO(response.audio_content), "mp3")
            with threading.Lock():
                current_playback = play(audio)
            current_playback.wait_done()
        tts_queue.task_done()

def text_to_speech_stream(text, lang_code):
    # Split text into natural speech chunks
    chunks = re.split(r'(?<=[.!?]) +', text)
    for chunk in chunks:
        if chunk.strip():
            tts_queue.put((chunk, lang_code))

def chat():
    # Start TTS worker thread
    threading.Thread(target=tts_worker, daemon=True).start()
    
    print("\n🤖 AI Chatbot - Type 'exit' to quit")
    initial_text = "Hello! Welcome to TATA Motors. How can I assist you today?"
    lang = detect_language(initial_text)
    text_to_speech_stream(initial_text, lang)
    print(f"\n🤖 AI: {initial_text}")

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() == 'exit':
            break
            
        # Interrupt current playback
        interrupt_flag.set()
        with threading.Lock():
            if current_playback and current_playback.is_playing():
                current_playback.stop()
        tts_queue.queue.clear()
        interrupt_flag.clear()
        
        # Stream Gemini response
        full_response = []
        lang = detect_language(user_input)
        for chunk in stream_gemini_response(user_input):
            print(chunk, end='', flush=True)
            full_response.append(chunk)
            text_to_speech_stream(chunk, lang)
        
        print(f"\n🤖 AI: {''.join(full_response)}")
    
    tts_queue.put((None, None))  # Stop worker

if __name__ == "__main__":
    chat()