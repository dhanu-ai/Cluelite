# stt_realtime_ws.py
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
SAMPLERATE = 16000   # Standard voice rate
CHUNK_DURATION = 5   # seconds
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)
BLOCKSIZE = 1024
MAX_BUFFER_SIZE = SAMPLERATE * 10  # keep last 10 seconds only
TRANSCRIPT_TRIM_LIMIT = 3000       # chars
TRANSCRIPT_KEEP_AFTER_TRIM = 2000  # chars

# Environment overrides
ENV_MIC = os.getenv("STT_MIC_INDEX")
ENV_SPK = os.getenv("STT_SPEAKER_INDEX")
ALLOW_VB_CABLE = os.getenv("ALLOW_VB_CABLE", "0").lower() in ("1", "true", "yes")

# === Technical Term Mapping ===
TECHNICAL_TERMS = {
    "core temperature": "conversion formula",
    "yellow": "output",
    "select this converter": "analyze this function",
    "faranheit": "fahrenheit",
    "celcius": "celsius",
    # Add or extend your mapping here...
}

# === Shared Buffers and State ===
mic_buffer = np.zeros((0,), dtype=np.float32)
speaker_buffer = np.zeros((0,), dtype=np.float32)
mic_queue = queue.Queue(maxsize=5)
speaker_queue = queue.Queue(maxsize=5)
latest_transcripts = {"interviewee": "", "interviewer": ""}
lock = threading.Lock()
stream_active = False

# device indexes in use (may change via hot-reload)
current_mic_index = None
current_speaker_index = None
_device_change_flag = threading.Event()

# === Device helpers ===
def log_audio_devices():
    devices = sd.query_devices()
    logger.info("=== Available Audio Devices ===")
    for i, dev in enumerate(devices):
        logger.info(f"Device {i}: {dev['name']} - Inputs: {dev['max_input_channels']}, Outputs: {dev['max_output_channels']}")
    logger.info("==============================")

def _normalize(name: str) -> str:
    return (name or "").lower().strip()

def _is_vb_cable(name: str) -> bool:
    n = _normalize(name)
    return any(k in n for k in ["vb-audio", "cable", "virtual cable", "vb audio point", "point"])

_MIC_PRI_KEYWORDS = [
    ("headset", 1000),
    ("headphones", 950),
    ("microphone array", 900),
    ("array", 850),
    ("microphone", 800),
]

_LOOPBACK_PRI_KEYWORDS = [
    ("stereo mix", 1000),
    ("what u hear", 980),
    ("loopback", 950),
    ("monitor of", 900),
    ("mix", 850),
]

_NEGATIVE_MIC = ["microsoft sound mapper", "vb-audio", "cable", "virtual", "point"]
_NEGATIVE_LOOP = ["vb-audio", "cable", "virtual", "point"]

def score_device(i, dev, kind: str):
    name = _normalize(dev["name"]) if isinstance(dev, dict) else _normalize(dev.name)
    inputs = dev["max_input_channels"] if isinstance(dev, dict) else dev.max_input_channels
    if inputs <= 0:
        return -10_000
    score = 0
    if kind == "mic":
        if any(bad in name for bad in _NEGATIVE_MIC):
            score -= 1000
        for kw, pts in _MIC_PRI_KEYWORDS:
            if kw in name:
                score += pts
    else:
        if any(bad in name for bad in _NEGATIVE_LOOP):
            score -= 1200
        for kw, pts in _LOOPBACK_PRI_KEYWORDS:
            if kw in name:
                score += pts
    if "realtek" in name:
        score += 50
    if "microsoft sound mapper" in name:
        score -= 2000
    score -= i * 0.1
    return score

def find_best_device(kind: str = "mic"):
    """Return best device index for kind == 'mic' or 'loopback' using scoring and env overrides."""
    # env override
    if kind == "mic" and ENV_MIC is not None:
        try:
            return int(ENV_MIC)
        except Exception:
            pass
    if kind == "loopback" and ENV_SPK is not None:
        try:
            return int(ENV_SPK)
        except Exception:
            pass

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # Prefer WASAPI headphone loopback for loopback kind
    if kind == "loopback":
        for i, dev in enumerate(devices):
            name = _normalize(dev["name"])
            try:
                host_name = hostapis[dev["hostapi"]]["name"].lower()
            except Exception:
                host_name = ""
            # device must be input-capable to open loopback in InputStream (WASAPI presents loopback as input)
            if dev["max_input_channels"] > 0 and "headphone" in name and "wasapi" in host_name:
                logger.info(f"Found WASAPI loopback headphone device: {dev['name']} (index {i})")
                return i

    # Score devices
    best_idx = None
    best_score = -10_000
    for i, dev in enumerate(devices):
        try:
            s = score_device(i, dev, kind)
        except Exception:
            s = -10_000
        if s > best_score:
            best_score = s
            best_idx = i

    if best_idx is None:
        logger.warning(f"No {kind} device found dynamically")
        return None

    picked = devices[best_idx]
    if _is_vb_cable(picked["name"]) and not ALLOW_VB_CABLE:
        # try next best non-vb
        candidates = [(i, score_device(i, d, kind)) for i, d in enumerate(devices) if not _is_vb_cable(d["name"]) and d["max_input_channels"] > 0]
        if candidates:
            best_idx = max(candidates, key=lambda x: x[1])[0]
        else:
            logger.warning(f"No non-VB-Cable {kind} device available; falling back to VB-Cable.")

    return best_idx

# === Audio Callbacks ===
def mic_callback(indata, frames, time, status):
    global mic_buffer
    if status.input_overflow:
        logger.warning("[Mic] Overflow")
        return
    mic_buffer = np.concatenate((mic_buffer, indata[:, 0]))
    if len(mic_buffer) > MAX_BUFFER_SIZE:
        mic_buffer = mic_buffer[-MAX_BUFFER_SIZE:]
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
    # In WASAPI loopback the input is stereo; convert to mono
    if indata.ndim > 1:
        mono = np.mean(indata, axis=1)
    else:
        mono = indata
    speaker_buffer = np.concatenate((speaker_buffer, mono))
    if len(speaker_buffer) > MAX_BUFFER_SIZE:
        speaker_buffer = speaker_buffer[-MAX_BUFFER_SIZE:]
    if len(speaker_buffer) >= CHUNK_SIZE:
        try:
            speaker_queue.put_nowait(speaker_buffer[:CHUNK_SIZE])
        except queue.Full:
            logger.debug("[Speaker] Queue full - skipping")
        speaker_buffer = speaker_buffer[CHUNK_SIZE:]

# === Post-processing ===
def post_process_transcript(text):
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
                time.sleep(0.5)
            if text:
                text = post_process_transcript(text)
                with lock:
                    current = latest_transcripts[role]
                    if len(current) > TRANSCRIPT_TRIM_LIMIT:
                        latest_transcripts[role] = current[-TRANSCRIPT_KEEP_AFTER_TRIM:] + " " + text
                    else:
                        latest_transcripts[role] += " " + text
                logger.info(f"[{role.upper()}] {text}")
        except Exception as e:
            logger.exception(f"[{role}] STT error: {e}")

# === Device monitor (hot-reload) ===
class DeviceMonitor(threading.Thread):
    def __init__(self, poll_interval=2.0):
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.running = True
        self._last_default_output = None

    def get_default_output_index(self):
        try:
            default = sd.default.device
            if isinstance(default, (list, tuple)) and len(default) >= 2:
                return default[1]
            return None
        except Exception:
            return None

    def run(self):
        global current_mic_index, current_speaker_index
        while self.running:
            try:
                out_idx = self.get_default_output_index()
                if out_idx != self._last_default_output:
                    self._last_default_output = out_idx
                    logger.info(f"Default output changed (index={out_idx})")
                    # pick devices again (respect env overrides)
                    mic_idx = find_best_device("mic")
                    spk_idx = None
                    # attempt WASAPI loopback for the current default output (if possible)
                    try:
                        devices = sd.query_devices()
                        hostapis = sd.query_hostapis()
                        if out_idx is not None and out_idx < len(devices):
                            out_dev = devices[out_idx]
                            out_name = _normalize(out_dev.get("name", ""))
                            # search for WASAPI loopback entry matching this output device name
                            for i, dev in enumerate(devices):
                                if dev["max_input_channels"] <= 0:
                                    continue
                                host_name = hostapis[dev["hostapi"]]["name"].lower() if dev.get("hostapi") is not None else ""
                                if "wasapi" in host_name and "headphone" in out_name and "headphone" in _normalize(dev.get("name","")):
                                    spk_idx = i
                                    logger.info(f"Selected WASAPI loopback device {dev['name']} (idx {i}) for output {out_dev['name']}")
                                    break
                    except Exception:
                        spk_idx = None

                    if spk_idx is None:
                        spk_idx = find_best_device("loopback")

                    # apply new picks
                    changed = False
                    if mic_idx is not None and mic_idx != current_mic_index:
                        current_mic_index = mic_idx
                        changed = True
                    if spk_idx is not None and spk_idx != current_speaker_index:
                        current_speaker_index = spk_idx
                        changed = True

                    if changed:
                        logger.info(f"DeviceMonitor applied new devices: mic={current_mic_index}, speaker={current_speaker_index}")
                        _device_change_flag.set()
            except Exception as e:
                logger.exception(f"Device monitor error: {e}")
            time.sleep(self.poll_interval)

    def stop(self):
        self.running = False

# === Public API ===
def get_latest_transcripts():
    with lock:
        return latest_transcripts.copy()

def clear_transcripts():
    with lock:
        latest_transcripts["interviewee"] = ""
        latest_transcripts["interviewer"] = ""

def is_stt_ready():
    return stream_active

# === Main streaming loop ===
def start_stt_streams():
    global stream_active, current_mic_index, current_speaker_index
    if stream_active:
        logger.warning("STT streams already active")
        return True

    log_audio_devices()

    # initial picks
    current_mic_index = find_best_device("mic")
    current_speaker_index = find_best_device("loopback")

    if current_mic_index is None or current_speaker_index is None:
        logger.error("❌ Cannot start STT: mic or speaker not found")
        return False

    logger.info(f"Using mic device index: {current_mic_index} ({sd.query_devices()[current_mic_index]['name']})")
    logger.info(f"Using speaker device index: {current_speaker_index} ({sd.query_devices()[current_speaker_index]['name']})")

    # start workers for transcription uploads
    threading.Thread(target=stt_worker, args=("interviewee", mic_queue, latest_transcripts, lock), daemon=True).start()
    threading.Thread(target=stt_worker, args=("interviewer", speaker_queue, latest_transcripts, lock), daemon=True).start()

    # start device monitor
    monitor = DeviceMonitor()
    monitor.start()

    def stream_loop():
        global stream_active, current_mic_index, current_speaker_index
        try:
            stream_active = True
            while stream_active:
                _device_change_flag.clear()
                mic_idx = current_mic_index
                spk_idx = current_speaker_index

                # We'll attempt to open two InputStreams:
                #  - mic InputStream (normal)
                #  - speaker InputStream: first try WASAPI loopback on the output device if available; 
                #    if not possible, open the selected loopback device (Stereo Mix etc.)
                try:
                    # mic stream - normal input
                    mic_stream = sd.InputStream(device=mic_idx, channels=1, samplerate=SAMPLERATE,
                                                blocksize=BLOCKSIZE, callback=mic_callback, dtype='float32')
                    # speaker stream - try WASAPI loopback first if the chosen speaker index corresponds to an output device with WASAPI support
                    speaker_stream = None
                    opened_wasapi_loopback = False

                    # attempt WASAPI loopback: find an output device index to loopback from that matches current_speaker_index name or default output
                    try:
                        hostapis = sd.query_hostapis()
                        devices = sd.query_devices()

                        # If the selected spk_idx is actually a "loopback" presented input (like Stereo Mix), we still use it.
                        # But prefer WASAPI loopback from default output if available:
                        default_out = None
                        try:
                            default_dev = sd.default.device
                            if isinstance(default_dev, (list, tuple)) and len(default_dev) >= 2:
                                default_out = default_dev[1]
                        except Exception:
                            default_out = None

                        # Candidate output indices (prefer default_out, then any device with "headphone" in the name)
                        candidates_output = []
                        if default_out is not None:
                            candidates_output.append(default_out)
                        for i, d in enumerate(devices):
                            if d["max_output_channels"] > 0 and "headphone" in _normalize(d["name"]):
                                if i not in candidates_output:
                                    candidates_output.append(i)

                        # Try to open a WASAPI loopback InputStream using candidate output devices
                        for out_idx in candidates_output:
                            try:
                                out_dev = devices[out_idx]
                                host_name = hostapis[out_dev["hostapi"]]["name"].lower() if out_dev.get("hostapi") is not None else ""
                                if "wasapi" not in host_name:
                                    continue
                                # Attempt WASAPI loopback by opening an InputStream with extra_settings
                                wasapi_settings = sd.WasapiSettings(loopback=True)
                                speaker_stream = sd.InputStream(device=out_idx, channels=2, samplerate=SAMPLERATE,
                                                               blocksize=BLOCKSIZE, callback=speaker_callback,
                                                               dtype='float32', extra_settings=wasapi_settings)
                                # Try to start stream to ensure it works
                                speaker_stream.start()
                                opened_wasapi_loopback = True
                                logger.info(f"Opened WASAPI loopback on output device {out_dev['name']} (index {out_idx})")
                                # update chosen speaker index so monitor knows this is in use
                                current_speaker_index = out_idx
                                break
                            except Exception as e:
                                # failed to open this wasapi loopback candidate, try next
                                if speaker_stream is not None:
                                    try:
                                        speaker_stream.close()
                                    except Exception:
                                        pass
                                speaker_stream = None
                                opened_wasapi_loopback = False
                                logger.debug(f"WASAPI loopback attempt failed for out_idx {out_idx}: {e}")
                    except Exception as e:
                        logger.debug(f"WASAPI detection error: {e}")
                        opened_wasapi_loopback = False
                        speaker_stream = None

                    # If we couldn't open a WASAPI loopback, fall back to the chosen loopback device (Stereo Mix etc.)
                    if not opened_wasapi_loopback:
                        if speaker_stream is not None:
                            try:
                                speaker_stream.close()
                            except Exception:
                                pass
                            speaker_stream = None
                        # open chosen loopback device (this will typically be Stereo Mix or Virtual Cable input)
                        speaker_stream = sd.InputStream(device=spk_idx, channels=2, samplerate=SAMPLERATE,
                                                       blocksize=BLOCKSIZE, callback=speaker_callback, dtype='float32')

                    # Open mic stream and speaker stream within context managers (so they close cleanly on change)
                    with mic_stream, speaker_stream:
                        logger.info("🎙️ STT streams started (mic + speaker capture)")
                        # stay in stream until device change or stop requested
                        while stream_active and not _device_change_flag.is_set():
                            time.sleep(0.25)

                        if _device_change_flag.is_set():
                            logger.info("Device change detected — restarting streams with new devices")
                            # streams will be closed by context managers; loop will restart and re-open
                            continue

                except Exception as e:
                    logger.error(f"Stream error (opening streams): {e}")
                    # wait and retryKO
                    time.sleep(1.0)
        finally:
            stream_active = False
            try:
                monitor.stop()
            except Exception:
                pass

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

if __name__ == "__main__":
    # Quick manual test
    start_stt_streams()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_stt_streams()
        logger.info("Exited by user")
