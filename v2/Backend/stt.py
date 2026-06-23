# stt.py
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
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cluelite.STT")

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("AssemblyAI API key missing")

HEADERS = {"authorization": ASSEMBLYAI_API_KEY}
SAMPLERATE = 16000
CHUNK_DURATION = 3  # Increased from 2 to reduce frequency
CHUNK_SIZE = SAMPLERATE * CHUNK_DURATION
BLOCKSIZE = 2048  # Increased block size

mic_buffer = np.zeros((0,), dtype=np.float32)
speaker_buffer = np.zeros((0,), dtype=np.float32)
mic_queue = queue.Queue(maxsize=3)  # Reduced queue size to prioritize latest data
speaker_queue = queue.Queue(maxsize=3)  # Reduced queue size
latest_transcripts = {"interviewee": "", "interviewer": ""}
lock = threading.Lock()
stream_active = False
processing_paused = False

# === Audio callbacks ===
def mic_callback(indata, frames, time, status):
    global mic_buffer
    if status.input_overflow: 
        return
    if processing_paused:
        return
        
    mic_buffer = np.concatenate((mic_buffer, indata[:,0]))
    if len(mic_buffer) >= CHUNK_SIZE:
        # Clear queue if full and add latest data
        while mic_queue.qsize() >= 2:
            try:
                mic_queue.get_nowait()
            except queue.Empty:
                break
        try:
            mic_queue.put_nowait(mic_buffer[:CHUNK_SIZE].copy())
        except queue.Full:
            pass
        mic_buffer = mic_buffer[CHUNK_SIZE:]

def speaker_callback(indata, frames, time, status):
    global speaker_buffer
    if status.input_overflow: 
        return
    if processing_paused:
        return
        
    mono = np.mean(indata, axis=1)
    speaker_buffer = np.concatenate((speaker_buffer, mono))
    if len(speaker_buffer) >= CHUNK_SIZE:
        # Clear queue if full and add latest data
        while speaker_queue.qsize() >= 2:
            try:
                speaker_queue.get_nowait()
            except queue.Empty:
                break
        try:
            speaker_queue.put_nowait(speaker_buffer[:CHUNK_SIZE].copy())
        except queue.Full:
            pass
        speaker_buffer = speaker_buffer[CHUNK_SIZE:]

# === STT Worker ===
def stt_worker(role: str, q: queue.Queue, latest_transcripts: Dict[str, str], lock: threading.Lock):
    consecutive_errors = 0
    max_errors = 5
    
    while True:
        try:
            audio = q.get()
            if audio is None:
                break
                
            # Skip processing if queue is backing up
            if q.qsize() > 1:
                logger.debug(f"[{role}] Skipping processing due to queue backup")
                continue
                
            buffer = io.BytesIO()
            sf.write(buffer, audio, SAMPLERATE, format="WAV")
            buffer.seek(0)
            
            upload_resp = requests.post(
                "https://api.assemblyai.com/v2/upload", 
                headers=HEADERS, 
                data=buffer,
                timeout=15
            )
            
            if upload_resp.status_code != 200:
                logger.warning(f"[{role}] Audio upload failed: {upload_resp.status_code}")
                continue
                
            audio_url = upload_resp.json().get("upload_url")
            
            transcript_resp = requests.post(
                "https://api.assemblyai.com/v2/transcript", 
                headers=HEADERS, 
                json={
                    "audio_url": audio_url,
                    "language_detection": True
                },
                timeout=15
            )
            
            if transcript_resp.status_code != 200:
                logger.warning(f"[{role}] Transcription request failed: {transcript_resp.status_code}")
                continue
                
            transcript_id = transcript_resp.json().get("id")
            text = ""
            
            # Poll for results with shorter timeout
            for _ in range(20):
                try:
                    poll_resp = requests.get(
                        f"https://api.assemblyai.com/v2/transcript/{transcript_id}", 
                        headers=HEADERS,
                        timeout=5
                    )
                    
                    if poll_resp.status_code != 200:
                        break
                        
                    status = poll_resp.json().get("status")
                    if status == "completed": 
                        text = poll_resp.json().get("text", "").strip()
                        break
                    elif status == "error":
                        logger.warning(f"[{role}] Transcription error")
                        break
                        
                    time.sleep(0.5)
                except requests.exceptions.RequestException:
                    break
                    
            if text:
                with lock:
                    timestamp = time.strftime("[%H:%M:%S] ")
                    latest_transcripts[role] += f"\n{timestamp}{text}"
                    
                    # Limit transcript history
                    lines = latest_transcripts[role].split('\n')
                    if len(lines) > 20:
                        latest_transcripts[role] = '\n'.join(lines[-20:])
                        
                logger.info(f"[{role.upper()}] {text}")
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    logger.warning(f"[{role}] Too many consecutive errors, pausing for 10s")
                    time.sleep(10)
                    consecutive_errors = 0
                
        except Exception as e:
            logger.exception(f"[{role}] STT error: {e}")
            time.sleep(2)

# === Public API ===
def get_latest_transcripts():
    with lock: 
        return latest_transcripts.copy()

def clear_transcripts():
    with lock: 
        latest_transcripts["interviewee"] = ""
        latest_transcripts["interviewer"] = ""

def pause_processing():
    global processing_paused
    processing_paused = True

def resume_processing():
    global processing_paused
    processing_paused = False

def start_stt_streams() -> bool:
    global stream_active, processing_paused
    if stream_active:
        return True
        
    try:
        processing_paused = False
        
        devices = sd.query_devices()
        mic_index = None
        speaker_index = None
        
        for i, d in enumerate(devices):
            name = d["name"].lower()
            if mic_index is None and ("mic" in name or "microphone" in name):
                mic_index = i
            if speaker_index is None and ("stereo mix" in name or "loopback" in name or "output" in name):
                speaker_index = i
        
        if mic_index is None:
            mic_index = sd.default.device[0]
        if speaker_index is None:
            speaker_index = sd.default.device[1] if len(sd.default.device) > 1 else sd.default.device[0]
            
        if mic_index is None or speaker_index is None:
            logger.error("Could not find suitable audio devices")
            return False
        
        threading.Thread(
            target=stt_worker, 
            args=("interviewee", mic_queue, latest_transcripts, lock), 
            daemon=True,
            name=f"stt_worker_interviewee"
        ).start()
        
        threading.Thread(
            target=stt_worker, 
            args=("interviewer", speaker_queue, latest_transcripts, lock), 
            daemon=True,
            name=f"stt_worker_interviewer"
        ).start()
        
        def stream_loop():
            global stream_active
            stream_active = True
            
            try:
                with sd.InputStream(
                    device=mic_index, 
                    channels=1, 
                    samplerate=SAMPLERATE, 
                    blocksize=BLOCKSIZE, 
                    callback=mic_callback
                ), sd.InputStream(
                    device=speaker_index, 
                    channels=2, 
                    samplerate=SAMPLERATE, 
                    blocksize=BLOCKSIZE, 
                    callback=speaker_callback
                ):
                    while stream_active: 
                        sd.sleep(1000)
            except Exception as e:
                logger.error(f"Audio stream error: {e}")
                stream_active = False
        
        threading.Thread(target=stream_loop, daemon=True, name="audio_stream_loop").start()
        return True
        
    except Exception as e:
        logger.error(f"Failed to start STT streams: {e}")
        return False

def stop_stt_streams():
    global stream_active, processing_paused
    stream_active = False
    processing_paused = True
    
    try:
        mic_queue.put(None, timeout=1)
        speaker_queue.put(None, timeout=1)
    except queue.Full:
        pass
        
    while not mic_queue.empty():
        try:
            mic_queue.get_nowait()
        except queue.Empty:
            break
            
    while not speaker_queue.empty():
        try:
            speaker_queue.get_nowait()
        except queue.Empty:
            break