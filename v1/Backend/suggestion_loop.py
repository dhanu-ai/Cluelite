import time
import threading
import logging
from stt import get_latest_transcripts
from ocr import get_latest_ocr_text
from groq_ai import generate_suggestion

logger = logging.getLogger("Cluelite")

latest_suggestion = ""
suggestion_lock = threading.Lock()
LAST_SUGGESTION_TIME = 0
last_combined_prompt = ""

def get_latest_suggestion():
    with suggestion_lock:
        return latest_suggestion

def run_suggestion_loop(stop_flag):
    global latest_suggestion, LAST_SUGGESTION_TIME, last_combined_prompt
    logger.info("📡 [Suggestion Loop] Running...")

    while not stop_flag.is_set():
        try:
            transcripts = get_latest_transcripts()
            ocr_lines = get_latest_ocr_text()
            ocr_text = " | ".join(ocr_lines)
            
            if not transcripts["interviewee"].strip() and not ocr_text.strip():
                time.sleep(1)
                continue
                
            current_time = time.time()
            if current_time - LAST_SUGGESTION_TIME < 5:
                time.sleep(1)
                continue
                
            current_prompt = f"{transcripts['interviewee']}|{ocr_text}"
            if current_prompt == last_combined_prompt:
                time.sleep(1)
                continue
                
            last_combined_prompt = current_prompt
            combined_prompt = f"""
[INTERVIEWEE]: {transcripts['interviewee']}
[SCREEN]: {ocr_text}
"""

            suggestion = generate_suggestion(
                transcript=combined_prompt,
                ocr_text=ocr_text,
                timeout=5
            )

            with suggestion_lock:
                latest_suggestion = suggestion

            logger.info(f"✅ [SUGGESTION] Updated: {suggestion[:80]}{'...' if len(suggestion) > 80 else ''}")
            LAST_SUGGESTION_TIME = current_time

        except Exception as e:
            logger.error(f"❌ Suggestion loop error: {e}")

        time.sleep(1)
    logger.info("🛑 Suggestion loop stopped.")