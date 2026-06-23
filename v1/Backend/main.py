from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import logging
import tempfile
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Cluelite")

# === Core Modules ===
from ocr import ocr_loop, get_latest_ocr_text, clear_ocr_history
from stt import get_latest_transcripts, start_stt_streams, stop_stt_streams, clear_transcripts
from groq_ai import generate_suggestion
from resend_mailer import send_email
from suggestion_loop import run_suggestion_loop, get_latest_suggestion

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global State ===
stop_flag = threading.Event()
threads_started = False

# === Routes ===

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/start")
def start_monitoring():
    global threads_started, stop_flag
    if not threads_started:
        stop_flag.clear()
        clear_transcripts()
        clear_ocr_history()
        
        threading.Thread(target=ocr_loop, args=(stop_flag,), daemon=True).start()
        logger.info("✅ OCR thread started")

        threading.Thread(target=run_suggestion_loop, args=(stop_flag,), daemon=True).start()
        logger.info("✅ Suggestion loop started")

        start_stt_streams()
        logger.info("🎙️ STT streams started")

        threads_started = True
        logger.info("📈 Monitoring started")
    return {"status": "started"}

@app.get("/suggestion")
def get_suggestion():
    transcripts = get_latest_transcripts()
    ocr = get_latest_ocr_text()

    combined_transcript = f"{transcripts.get('interviewee', '')} {transcripts.get('interviewer', '')}"

    if not combined_transcript.strip() and not any(ocr):
        return {"suggestion": "No content available for suggestions"}

    suggestion = generate_suggestion(
        transcript=combined_transcript,
        ocr_text=" | ".join(ocr),
        mode="auto"
    )

    logger.info(f"🛰️ Serving Suggestion: {suggestion}")
    return {"suggestion": suggestion or "No suggestions yet..."}

@app.get("/debug")
async def debug_session():
    return {
        "ocr": get_latest_ocr_text(),
        "transcript": get_latest_transcripts(),
        "suggestion": get_latest_suggestion()
    }

@app.post("/stop")
async def stop_monitoring():
    global threads_started, stop_flag

    final_ocr = get_latest_ocr_text()
    final_transcripts = get_latest_transcripts()

    full_transcript = f"""
INTERVIEWER TRANSCRIPT:
{final_transcripts.get('interviewer', '')}

INTERVIEWEE TRANSCRIPT:
{final_transcripts.get('interviewee', '')}

SCREEN CONTENT:
{' | '.join(final_ocr)}
"""

    if not final_transcripts.get('interviewer', '') and not final_transcripts.get('interviewee', ''):
        full_transcript = "WARNING: No audio captured during session\n\n" + full_transcript

    suggestion = generate_suggestion(
        transcript=full_transcript,
        ocr_text="",
        mode="analysis"
    )

    stop_flag.set()
    stop_stt_streams()
    threads_started = False

    transcript_id = str(uuid.uuid4())
    transcript_file = f"transcript_{transcript_id}.txt"
    
    with open(transcript_file, "w") as f:
        f.write(full_transcript)
    
    logger.info("📨 Sending summary email")
    try:
        send_email(
            "Cluelite Interview Summary", 
            suggestion, 
            attachment=transcript_file
        )
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        return {"status": "Session ended but email failed"}
    
    try:
        os.remove(transcript_file)
    except Exception as e:
        logger.error(f"Error deleting transcript: {e}")

    return {"status": "Session ended. Mail sent."}