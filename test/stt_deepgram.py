import queue
import json
import pyaudio
import os
from dotenv import load_dotenv
import websocket
import openai
from threading import Thread

# Load API keys from .env
load_dotenv()
print("Deepgram API Key:", os.getenv("DEEPGRAM_API_KEY"))
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk

# OpenAI GPT response function
def get_gpt_response(prompt, language):
    messages = [{"role": "system", "content": "You are a multilingual assistant."},
                {"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# WebSocket event handlers
def on_message(ws, message):
    data = json.loads(message)
    if "channel" in data and "alternatives" in data["channel"]:
        transcript = data["channel"]["alternatives"][0].get("transcript", "")
        if transcript:
            print("User:", transcript)
            response = get_gpt_response(transcript, "auto")
            print("AI:", response)

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed:", close_status_code, close_msg)

def on_open(ws):
    def run():
        with MicrophoneStream(RATE, CHUNK) as stream:
            for audio_chunk in stream.generator():
                ws.send(audio_chunk, websocket.ABNF.OPCODE_BINARY)
        ws.close()
    Thread(target=run, daemon=True).start()

def main():
    language_code = "hi,ar,en"
    deepgram_url = f"wss://api.deepgram.com/v1/listen?access_token={DEEPGRAM_API_KEY}&model=nova&language={language_code}"
    
    ws = websocket.WebSocketApp(deepgram_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

if __name__ == "__main__":
    main()
