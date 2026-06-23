#stt file
import os
import io
import time
import queue
import logging
import threading
import requests
import numpy as np
import soundfile as sf
import sounddevice as sd

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Cluelite.STT")

# === Config ===
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    logger.error("AssemblyAI API key missing!")
    raise RuntimeError("AssemblyAI API key not configured")

HEADERS = {"authorization": ASSEMBLYAI_API_KEY}
SAMPLERATE = 16000  # Standard for voice
CHUNK_DURATION = 2
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)
BLOCKSIZE = 1024

# === Technical Term Mapping ===
TECHNICAL_TERMS = {
    "core temperature": "conversion formula",
    "yellow": "output",
    "select this converter": "analyze this function",
    "faranheit": "fahrenheit",
    "celcius": "celsius",
    "syntax": "syntax",
    "variable": "variable",
    "function": "function",
    "loop": "loop",
    "condition": "condition",
    "bug": "bug",
    "debug": "debug",
    "algorithm": "algorithm",
    "parameter": "parameter",
    "argument": "argument",
    "object": "object",
    "class": "class",
    "instance": "instance",
    "method": "method",
    "property": "property",
    "git": "git",
    "commit": "commit",
    "repository": "repository",
    "branch": "branch",
    "merge": "merge",
    "api": "API",
    "endpoint": "endpoint",
    "json": "JSON",
    "database": "database",
    "query": "query",
    "server": "server",
    "client": "client",
    "frontend": "frontend",
    "backend": "backend",
    "framework": "framework",
    "library": "library",
    "dependency": "dependency",
    "environment": "environment",
    "deployment": "deployment",
    "container": "container",
    "microservice": "microservice",
    "authentication": "authentication",
    "authorization": "authorization",
    "encryption": "encryption",
    "https": "HTTPS",
    "ssl": "SSL",
    "tls": "TLS"
}

# === Shared Buffers and Queues ===
mic_buffer = np.zeros((0,), dtype=np.float32)
speaker_buffer = np.zeros((0,), dtype=np.float32)
mic_queue = queue.Queue(maxsize=5)
speaker_queue = queue.Queue(maxsize=5)
latest_transcripts = {"interviewee": "", "interviewer": ""}
lock = threading.Lock()
stream_active = False

# === Device Verification ===
def log_audio_devices():
    devices = sd.query_devices()
    logger.info("=== Available Audio Devices ===")
    for i, dev in enumerate(devices):
        logger.info(f"Device {i}: {dev['name']} - "
                    f"Inputs: {dev['max_input_channels']}, "
                    f"Outputs: {dev['max_output_channels']}")
    logger.info("==============================")

# === Device Selectors ===
def find_device(kind="mic"):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if kind == "mic" and dev["max_input_channels"] > 0 and ("mic" in name or "microphone" in name):
            return i
        if kind == "loopback" and dev["max_input_channels"] > 0 and ("stereo mix" in name or "loopback" in name or "cable output" in name):
            return i
    logger.warning(f"No {kind} device found")
    return None

# === Audio Callbacks ===
def mic_callback(indata, frames, time, status):
    global mic_buffer
    if status.input_overflow:
        logger.warning("[Mic] Overflow")
        return
    mic_buffer = np.concatenate((mic_buffer, indata[:, 0]))
    if len(mic_buffer) >= CHUNK_SIZE:
        try:
            mic_queue.put_nowait(mic_buffer[:CHUNK_SIZE])
        except queue.Full:
            logger.debug("[Mic] Queue full - skipping")
        mic_buffer = mic_buffer[CHUNK_SIZE:]

def speaker_callback(indata, frames, time, status):
    global speaker_buffer
    if status.input_overflow:
        logger.warning("[Speaker] Overflow")
        return
    mono = np.mean(indata, axis=1)
    speaker_buffer = np.concatenate((speaker_buffer, mono))
    if len(speaker_buffer) >= CHUNK_SIZE:
        try:
            speaker_queue.put_nowait(speaker_buffer[:CHUNK_SIZE])
        except queue.Full:
            logger.debug("[Speaker] Queue full - skipping")
        speaker_buffer = speaker_buffer[CHUNK_SIZE:]

# === STT Post-Processing ===
def post_process_transcript(text):
    """Correct common misrecognitions of technical terms"""
    for wrong, correct in TECHNICAL_TERMS.items():
        text = text.replace(wrong, correct)
    return text

# === STT Worker ===
def stt_worker(role, q, latest_transcripts, lock, samplerate=16000, max_poll_attempts=60):
    while True:
        audio = q.get()
        if audio is None:
            break

        try:
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate, format="WAV")
            buffer.seek(0)

            upload_resp = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=HEADERS,
                data=buffer
            )

            if upload_resp.status_code != 200:
                logger.error(f"[{role}] Upload failed: {upload_resp.text}")
                continue

            audio_url = upload_resp.json().get("upload_url")

            transcript_resp = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=HEADERS,
                json={"audio_url": audio_url}
            )

            if transcript_resp.status_code != 200:
                logger.error(f"[{role}] Transcript request failed: {transcript_resp.text}")
                continue

            transcript_id = transcript_resp.json().get("id")
            text = ""
            
            for _ in range(max_poll_attempts):
                poll_resp = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=HEADERS
                )
                status = poll_resp.json().get("status")
                if status == "completed":
                    text = poll_resp.json().get("text", "").strip()
                    break
                elif status == "error":
                    logger.error(f"[{role}] Transcription failed: {poll_resp.json().get('error')}")
                    break
                time.sleep(1)

            if text:
                # Apply technical term corrections
                text = post_process_transcript(text)
                
                with lock:
                    current = latest_transcripts[role]
                    if len(current) > 5000:
                        latest_transcripts[role] = current[-4000:] + " " + text
                    else:
                        latest_transcripts[role] += " " + text
                logger.info(f"[{role.upper()}] {text}")

        except Exception as e:
            logger.exception(f"[{role}] STT error: {e}")

# === Public API ===
def get_latest_transcripts():
    with lock:
        return latest_transcripts.copy()

def clear_transcripts():
    with lock:
        latest_transcripts["interviewee"] = ""
        latest_transcripts["interviewer"] = ""

def start_stt_streams():
    global stream_active
    if stream_active:
        logger.warning("STT streams already active")
        return True
        
    log_audio_devices()
    
    mic_index = find_device("mic")
    speaker_index = find_device("loopback")
    
    if mic_index is None or speaker_index is None:
        logger.error("❌ Cannot start STT: mic or speaker not found")
        return False
        
    logger.info(f"Using mic device index: {mic_index}")
    logger.info(f"Using speaker device index: {speaker_index}")

    threading.Thread(
        target=stt_worker, 
        args=("interviewee", mic_queue, latest_transcripts, lock),
        daemon=True
    ).start()
    threading.Thread(
        target=stt_worker, 
        args=("interviewer", speaker_queue, latest_transcripts, lock),
        daemon=True
    ).start()

    def stream_loop():
        global stream_active
        try:
            stream_active = True
            with sd.InputStream(device=mic_index, channels=1, samplerate=SAMPLERATE, 
                               blocksize=BLOCKSIZE, callback=mic_callback), \
                 sd.InputStream(device=speaker_index, channels=2, samplerate=SAMPLERATE, 
                               blocksize=BLOCKSIZE, callback=speaker_callback):
                logger.info("🎙️ STT streams started")
                while stream_active:
                    sd.sleep(500)
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            stream_active = False

    threading.Thread(target=stream_loop, daemon=True).start()
    return True

def stop_stt_streams():
    global stream_active
    try:
        stream_active = False
        mic_queue.put(None)
        speaker_queue.put(None)
        logger.info("🛑 STT streams stopped")
    except Exception as e:
        logger.error(f"Failed to stop STT streams: {e}")