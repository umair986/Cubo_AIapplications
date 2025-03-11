import os
import asyncio
import numpy as np
import sounddevice as sd
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import openai
from google.cloud import texttospeech

# Load API keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize APIs
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
openai.api_key = OPENAI_API_KEY

# Google TTS Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS
tts_client = texttospeech.TextToSpeechClient()

# Audio Config
SAMPLE_RATE = 16000  # Required by Deepgram
CHANNELS = 1  # Mono audio

# System Prompt for Chatbot
SYSTEM_PROMPT = "You are a multilingual AI assistant that responds in the same language as the user input. You can understand and respond in Hindi, Arabic, and English."

async def transcribe_audio():
    """Real-time Speech-to-Text (STT) and Chatbot interaction"""
    dg_connection = deepgram_client.listen.live.v("1")

    def on_transcript(data, **kwargs):
        if data and data.channel.alternatives:
            transcript = data.channel.alternatives[0].transcript
            if transcript:
                print(f"User: {transcript}")
                asyncio.create_task(process_chatbot_response(transcript))

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)

    options = LiveOptions(model="nova-3", language="en")  # Change language settings if needed
    dg_connection.start(options)

    print("🎤 Start speaking... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16, callback=lambda indata, *args: dg_connection.send(indata)):
        while True:
            await asyncio.sleep(0.1)  # Keep loop running

async def process_chatbot_response(user_input):
    """Process chatbot response & convert it to speech"""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )

    bot_response = response["choices"][0]["message"]["content"]
    print(f"🤖 Bot: {bot_response}")

    # Detect Language
    language_code = detect_language(bot_response)
    await speak_text(bot_response, language_code)

def detect_language(text):
    """Detect language based on common words (Basic Approach)"""
    if any(word in text for word in ["مرحبا", "كيف", "أنت"]):
        return "ar-XA"  # Arabic
    elif any(word in text for word in ["नमस्ते", "कैसे", "हो"]):
        return "hi-IN"  # Hindi
    else:
        return "en-US"  # Default to English

async def speak_text(text, language_code):
    """Convert text to speech using Google WaveNet"""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Play the audio response
    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
    sd.play(audio_data, samplerate=SAMPLE_RATE)
    sd.wait()

# Run the transcription loop
asyncio.run(transcribe_audio())
