import time
import threading
import logging
from difflib import SequenceMatcher

from stt import get_latest_transcripts, is_stt_ready
from ocr import get_latest_ocr_text
from groq_ai import generate_suggestion

logger = logging.getLogger("Cluelite")

# Shared state
latest_suggestion = ""
suggestion_lock = threading.Lock()
LAST_SUGGESTION_TIME = 0
last_combined_prompt = ""

# ===== Helper =====
def is_similar(text1, text2, threshold=0.9):
    """Check if two texts are similar (semantic duplicate filter)."""
    if not text1 or not text2:
        return False
    ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio >= threshold

# ===== Public getter =====
def get_latest_suggestion():
    with suggestion_lock:
        return latest_suggestion

# ===== Suggestion loop =====
def run_suggestion_loop(stop_flag):
    global latest_suggestion, LAST_SUGGESTION_TIME, last_combined_prompt
    logger.info("📡 [Suggestion Loop] Initializing...")

    # Wait for STT readiness
    logger.info("⏳ Waiting for STT service to be ready...")
    waited = 0
    while not is_stt_ready() and waited < 30:
        time.sleep(1)
        waited += 1

    if not is_stt_ready():
        logger.error("❌ STT service did not become ready in 30s. Aborting suggestion loop.")
        return

    logger.info("🎙️ STT ready — starting Suggestion Loop")

    while not stop_flag.is_set():
        try:
            transcripts = get_latest_transcripts()
            ocr_lines = get_latest_ocr_text()
            ocr_text = " | ".join(ocr_lines)

            # Skip if both transcript and OCR are empty
            if not transcripts["interviewee"].strip() and not ocr_text.strip():
                time.sleep(1)
                continue

            current_time = time.time()
            # Enforce cooldown to prevent spamming API
            if current_time - LAST_SUGGESTION_TIME < 5:
                time.sleep(1)
                continue

            # Build prompt
            current_prompt = f"{transcripts['interviewee']}|{ocr_text}".strip()

            # Skip if duplicate or very similar
            if is_similar(current_prompt, last_combined_prompt):
                logger.info("⏩ Skipping suggestion update - prompt unchanged or very similar")
                time.sleep(1)
                continue

            last_combined_prompt = current_prompt
            combined_prompt = f"""
[INTERVIEWEE]: {transcripts['interviewee']}
[SCREEN]: {ocr_text}
""".strip()

            # Generate suggestion
            suggestion = generate_suggestion(
                transcript=combined_prompt,
                ocr_text=ocr_text,
                timeout=5
            )

            # Store suggestion thread-safely
            with suggestion_lock:
                latest_suggestion = suggestion

            logger.info(f"✅ [SUGGESTION] Updated: {suggestion[:80]}{'...' if len(suggestion) > 80 else ''}")
            LAST_SUGGESTION_TIME = current_time

        except Exception as e:
            logger.error(f"❌ Suggestion loop error: {e}")

        time.sleep(1)

    logger.info("🛑 Suggestion loop stopped.")
