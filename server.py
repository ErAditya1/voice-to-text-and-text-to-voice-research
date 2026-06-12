import os
import json
import time
import asyncio
import numpy as np
import torch
import multiprocessing
import httpx
import io
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for models in stt-wishper first to avoid re-downloading
STT_WISHPER_DIR = os.environ.get("MODELS_DIR", "d:/Feeding_Trends/stt-wishper")
KOKORO_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"

if not os.path.exists(KOKORO_PATH) and os.path.exists(os.path.join(STT_WISHPER_DIR, KOKORO_PATH)):
    KOKORO_PATH = os.path.join(STT_WISHPER_DIR, KOKORO_PATH)
if not os.path.exists(VOICES_PATH) and os.path.exists(os.path.join(STT_WISHPER_DIR, VOICES_PATH)):
    VOICES_PATH = os.path.join(STT_WISHPER_DIR, VOICES_PATH)

# Global models
print("Loading Whisper model...")
num_cores = multiprocessing.cpu_count()
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=num_cores
)

print("Loading Silero VAD model...")
vad_model = load_silero_vad()

print("Loading Kokoro ONNX model...")
kokoro = Kokoro(KOKORO_PATH, VOICES_PATH)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512

# Warm up models to eliminate first-sentence delay
print("Warming up models...")
whisper_model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
try:
    # Warm up Kokoro
    for _ in kokoro.predict("warmup", voice="af_nicole", speed=1.0, lang="en-us"):
        pass
except Exception:
    pass
print("Warmup complete!")

# Serve static files from current directory
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/")
async def get():
    with open(os.path.join(BASE_DIR, "index.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

async def query_claude(api_key: str, model: str, prompt: str, history: list = None) -> str:
    """
    Queries the Anthropic Messages API with the provided prompt and optional history.
    """
    if not api_key:
        return "Error: Claude API Key is missing. Please enter it in the settings."
        
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Construct messages array
    messages = []
    if history:
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
    # Append current prompt
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # We want a concise response for voice conversation
    system_instruction = (
        "You are a helpful, extremely concise AI voice assistant. "
        "Keep your responses friendly, natural, and short (typically 1-3 sentences) "
        "so that they are easy to listen to. Do not use markdown styling like asterisks, "
        "bold text, or bullet points in your response."
    )
    
    payload = {
        "model": model,
        "max_tokens": 300,
        "system": system_instruction,
        "messages": messages
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", [])
                if content and len(content) > 0:
                    text_response = content[0].get("text", "").strip()
                    return text_response
                return "Error: Empty response received from Claude."
            else:
                return f"Error: Anthropic API returned status {response.status_code}. Detail: {response.text}"
    except Exception as e:
        return f"Error connecting to Anthropic API: {str(e)}"

@app.post("/api/claude/models")
async def get_claude_models(payload: dict):
    api_key = payload.get("api_key", "")
    if not api_key:
        return {"status": "error", "message": "API Key is required"}
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            response = await client.get("https://api.anthropic.com/v1/models", headers=headers)
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                models = []
                for m in models_data:
                    models.append({
                        "id": m.get("id"),
                        "name": m.get("display_name") or m.get("id")
                    })
                return {"status": "success", "models": models}
            else:
                return {"status": "error", "message": f"Anthropic API Error: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_engine_key(api_key, engine):
    if isinstance(api_key, dict):
        return api_key.get(engine, "")
    return api_key

async def process_pcm_stream(response_iterator):
    leftover = b""
    async for chunk in response_iterator:
        if chunk:
            chunk = leftover + chunk
            if len(chunk) % 2 != 0:
                leftover = chunk[-1:]
                chunk = chunk[:-1]
            else:
                leftover = b""
            if len(chunk) > 0:
                pcm_data = np.frombuffer(chunk, dtype=np.int16)
                float_data = pcm_data.astype(np.float32) / 32768.0
                yield float_data.tobytes()

async def stream_tts_audio(tts_engine, api_key, text, voice, speed):
    api_key = get_engine_key(api_key, tts_engine)
    if tts_engine == "elevenlabs" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=pcm_24000",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={"text": text, "model_id": "eleven_monolingual_v1"}
            )
            async for data in process_pcm_stream(response.aiter_bytes(chunk_size=2048)):
                yield data
                    
    elif tts_engine == "murf" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.murf.ai/v1/speech/generate",
                headers={"api-key": api_key, "Content-Type": "application/json", "Accept": "application/json"},
                json={
                    "voiceId": voice, "style": "Conversational", "text": text,
                    "rate": 0, "pitch": 0, "sampleRate": 24000,
                    "format": "WAV", "channelType": "MONO"
                }
            )
            if response.status_code == 200:
                audio_url = response.json().get("audioFile")
                if audio_url:
                    audio_resp = await client.get(audio_url)
                    wav_data = audio_resp.content
                    if len(wav_data) > 44:
                        pcm_bytes = wav_data[44:]
                        if len(pcm_bytes) % 2 != 0:
                            pcm_bytes = pcm_bytes[:-1]
                        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
                        float_data = pcm_data.astype(np.float32) / 32768.0
                        yield float_data.tobytes()
                        
    elif tts_engine == "openai" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "tts-1", "input": text, "voice": voice, "response_format": "pcm"}
            )
            async for data in process_pcm_stream(response.aiter_bytes(chunk_size=2048)):
                yield data
                    
    elif tts_engine == "deepgram" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&sample_rate=24000",
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                json={"text": text}
            )
            async for data in process_pcm_stream(response.aiter_bytes(chunk_size=2048)):
                yield data
                    
    elif tts_engine == "cartesia" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={"X-API-Key": api_key, "Cartesia-Version": "2024-06-10", "Content-Type": "application/json"},
                json={
                    "model_id": "sonic-english", "transcript": text, "voice": {"mode": "id", "id": voice},
                    "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 24000}
                }
            )
            async for chunk in response.aiter_bytes(chunk_size=2048):
                if chunk:
                    yield chunk
                    
    elif tts_engine == "fishaudio" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.fish.audio/v1/tts",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"text": text, "reference_id": voice, "format": "pcm"}
            )
            async for data in process_pcm_stream(response.aiter_bytes(chunk_size=2048)):
                yield data
                    
    elif tts_engine == "cosyvoice" and api_key:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.siliconflow.cn/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "FunAudioLLM/CosyVoice2-0.5B", "input": text, "voice": voice, "response_format": "pcm"}
            )
            if response.status_code == 200:
                async for data in process_pcm_stream(response.aiter_bytes(chunk_size=2048)):
                    yield data
            else:
                print(f"CosyVoice API Error: {response.status_code} - {response.text}")
                
    elif tts_engine == "local_cosyvoice" or tts_engine == "local_parakeet":
        print("Local Parakeet / CosyVoice is currently blocked by missing C++ Build Tools. Please install them.")
        yield np.zeros(2048, dtype=np.float32).tobytes()
                    
    else:
        stream = kokoro.create_stream(text, voice=voice, speed=speed, lang="en-us")
        async for samples, sample_rate in stream:
            yield samples.tobytes()

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            text = payload.get("text", "")
            voice = payload.get("voice", "af_nicole")
            speed = float(payload.get("speed", 1.0))
            tts_engine = payload.get("tts_engine", "local")
            api_key = payload.get("api_key", "")
            ai_conversation = payload.get("ai_conversation", False)
            claude_model = payload.get("claude_model", "claude-3-5-sonnet-20241022")
            
            if not text.strip():
                continue
                
            print(f"TTS Streaming requested for: '{text}' using voice '{voice}', engine '{tts_engine}'")
            start_time = time.time()
            first_chunk = True
            
            response_text = text
            llm_latency_ms = 0
            if ai_conversation:
                llm_start_time = time.time()
                claude_api_key = get_engine_key(api_key, "claude")
                response_text = await query_claude(claude_api_key, claude_model, text, None)
                llm_latency_ms = int((time.time() - llm_start_time) * 1000)
                await websocket.send_json({
                    "type": "ai_transcript",
                    "text": response_text
                })
            
            async for audio_bytes in stream_tts_audio(tts_engine, api_key, response_text, voice, speed):
                if first_chunk:
                    first_chunk = False
                    tts_gen_ms = int((time.time() - start_time) * 1000)
                    await websocket.send_json({
                        "type": "latency",
                        "tts_gen_ms": tts_gen_ms,
                        "llm_latency_ms": llm_latency_ms
                    })
                await websocket.send_bytes(audio_bytes)
                
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        print("TTS WebSocket disconnected")
    except Exception as e:
        print(f"Error in TTS WebSocket: {e}")

@app.websocket("/ws/s2s")
async def websocket_s2s(websocket: WebSocket):
    await websocket.accept()
    
    vad_iterator = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_silence_duration_ms=300
    )
    
    audio_buffer = []
    pre_speech_buffer = []
    max_pre_speech_chunks = 10
    is_speaking = False
    speech_end_time = 0.0
    voice = "af_nicole"
    speed = 1.0
    language = None
    task = "transcribe"
    stt_engine = "local"
    tts_engine = "local"
    api_key = ""
    ai_conversation = False
    claude_model = "claude-3-5-sonnet-20241022"
    chat_history = []
    
    try:
        while True:
            # Receive binary float32 PCM frames from the browser
            message = await websocket.receive()
            if "bytes" not in message:
                if "text" in message:
                    # Parse configurations sent as text
                    config = json.loads(message["text"])
                    voice = config.get("voice", voice)
                    speed = float(config.get("speed", speed))
                    lang_conf = config.get("language", "auto")
                    language = None if lang_conf == "auto" else lang_conf
                    task = config.get("task", "transcribe")
                    stt_engine = config.get("stt_engine", stt_engine)
                    tts_engine = config.get("tts_engine", tts_engine)
                    api_key = config.get("api_key", api_key)
                    ai_conversation = config.get("ai_conversation", ai_conversation)
                    claude_model = config.get("claude_model", claude_model)
                    continue
                continue
                
            chunk_bytes = message["bytes"]
            chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
            
            if len(chunk) == 0:
                continue
                
            chunk_tensor = torch.from_numpy(chunk)
            speech_dict = vad_iterator(chunk_tensor)
            
            if speech_dict:
                if "start" in speech_dict:
                    is_speaking = True
                    audio_buffer = list(pre_speech_buffer)
                    await websocket.send_json({"type": "status", "status": "speech_detected"})
                    
                if "end" in speech_dict:
                    speech_end_time = time.time()
                    is_speaking = False
                    await websocket.send_json({"type": "status", "status": "processing"})
                    
                    if audio_buffer:
                        full_audio = np.concatenate(audio_buffer, axis=0)
                        
                        def run_stt():
                            segs, _ = whisper_model.transcribe(
                                full_audio, 
                                beam_size=1,
                                language=language,
                                task=task
                            )
                            return " ".join(s.text for s in segs).strip()

                        async def run_elevenlabs_stt(audio_data, stt_key):
                            int_audio = (audio_data * 32767.0).astype(np.int16)
                            wav_io = io.BytesIO()
                            with wave.open(wav_io, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(SAMPLE_RATE)
                                wav_file.writeframes(int_audio.tobytes())
                            wav_io.seek(0)
                            
                            async with httpx.AsyncClient() as client:
                                response = await client.post(
                                    "https://api.elevenlabs.io/v1/speech-to-text",
                                    headers={"xi-api-key": stt_key},
                                    files={"file": ("audio.wav", wav_io, "audio/wav")},
                                    data={"model_id": "scribe_v1"}
                                )
                                return response.json().get("text", "")

                        async def run_deepgram_stt(audio_data, stt_key):
                            int_audio = (audio_data * 32767.0).astype(np.int16)
                            wav_io = io.BytesIO()
                            with wave.open(wav_io, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(SAMPLE_RATE)
                                wav_file.writeframes(int_audio.tobytes())
                            wav_bytes = wav_io.getvalue()
                            
                            async with httpx.AsyncClient() as client:
                                response = await client.post(
                                    "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                                    headers={
                                        "Authorization": f"Token {stt_key}",
                                        "Content-Type": "audio/wav"
                                    },
                                    content=wav_bytes
                                )
                                data = response.json()
                                if "results" in data and "channels" in data["results"] and len(data["results"]["channels"]) > 0:
                                    return data["results"]["channels"][0]["alternatives"][0]["transcript"]
                                return ""

                        loop = asyncio.get_running_loop()
                        stt_start_time = time.time()
                        
                        stt_api_key = get_engine_key(api_key, stt_engine)
                        if stt_engine == "elevenlabs" and stt_api_key:
                            text = await run_elevenlabs_stt(full_audio, stt_api_key)
                        elif stt_engine == "deepgram" and stt_api_key:
                            text = await run_deepgram_stt(full_audio, stt_api_key)
                        else:
                            text = await loop.run_in_executor(None, run_stt)
                            
                        stt_latency_ms = int((time.time() - stt_start_time) * 1000)
                        
                        if text:
                            await websocket.send_json({
                                "type": "transcript",
                                "text": text,
                                "stt_latency_ms": stt_latency_ms
                            })
                            
                            response_text = text
                            llm_latency_ms = 0
                            if ai_conversation:
                                llm_start = time.time()
                                claude_api_key = get_engine_key(api_key, "claude")
                                response_text = await query_claude(claude_api_key, claude_model, text, chat_history)
                                llm_latency_ms = int((time.time() - llm_start) * 1000)
                                
                                chat_history.append({"role": "user", "content": text})
                                chat_history.append({"role": "assistant", "content": response_text})
                                
                                # Cap the conversation history (last 20 turns)
                                if len(chat_history) > 40:
                                    chat_history = chat_history[-40:]
                                    
                                await websocket.send_json({
                                    "type": "ai_transcript",
                                    "text": response_text,
                                    "llm_latency_ms": llm_latency_ms
                                })
                            
                            # Stream back the synthesized reply
                            tts_start_time = time.time()
                            first_chunk = True
                            
                            async for audio_bytes in stream_tts_audio(tts_engine, api_key, response_text, voice, speed):
                                if first_chunk:
                                    first_chunk = False
                                    tts_gen_ms = int((time.time() - tts_start_time) * 1000)
                                    total_latency_ms = int((time.time() - speech_end_time) * 1000)
                                    await websocket.send_json({
                                        "type": "latency",
                                        "stt_latency_ms": stt_latency_ms,
                                        "llm_latency_ms": llm_latency_ms if ai_conversation else 0,
                                        "tts_gen_ms": tts_gen_ms,
                                        "total_latency_ms": total_latency_ms
                                    })
                                await websocket.send_bytes(audio_bytes)
                                
                            await websocket.send_json({"type": "done"})
                        else:
                            await websocket.send_json({"type": "transcript", "text": "[No speech recognized]"})
                            
                    audio_buffer = []
                    pre_speech_buffer = []
                    vad_iterator.reset_states()
                    
            if is_speaking:
                audio_buffer.append(chunk)
            else:
                pre_speech_buffer.append(chunk)
                if len(pre_speech_buffer) > max_pre_speech_chunks:
                    pre_speech_buffer.pop(0)
                    
    except WebSocketDisconnect:
        print("S2S WebSocket disconnected")
    except Exception as e:
        print(f"Error in S2S WebSocket: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
