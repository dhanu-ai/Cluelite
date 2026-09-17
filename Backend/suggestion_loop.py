# suggestion_loop.py
import time
import threading
import logging
from stt import get_latest_transcripts
from ocr import get_latest_ocr_text
from groq_ai import generate_suggestion
from typing import Dict
import hashlib
from queue import PriorityQueue
import queue

logger = logging.getLogger("Cluelite.suggestion_loop")
latest_suggestion = ""
suggestion_lock = threading.Lock()
LAST_SUGGESTION_TIME = 0
last_combined_prompt = ""
suggestion_interval = 5  # Increased from 2 to 5 seconds between suggestions

# Priority queue for critical content
critical_queue = PriorityQueue(maxsize=100)

def add_to_queue(item, priority=1):
    """Add item to priority queue."""
    try:
        critical_queue.put((priority, item), timeout=0.1)
    except queue.Full:
        logger.warning("Critical queue full, dropping item")

def get_latest_suggestion() -> str:
    """Get the latest suggestion with thread safety."""
    with suggestion_lock:
        return latest_suggestion

def compute_content_hash(transcripts, ocr_text):
    """Compute a hash of the content to detect changes."""
    content = f"{transcripts['interviewee']}|{transcripts['interviewer']}|{ocr_text}"
    return hashlib.md5(content.encode()).hexdigest()

def similarity(a, b):
    """Compute similarity between two strings (0-1)."""
    if not a or not b:
        return 0
        
    # Simple similarity measure based on common words
    a_words = set(a.split())
    b_words = set(b.split())
    
    if not a_words or not b_words:
        return 0
        
    intersection = a_words.intersection(b_words)
    union = a_words.union(b_words)
    
    return len(intersection) / len(union)

def run_suggestion_loop(stop_flag):
    """Main suggestion generation loop with improved logic."""
    global latest_suggestion, LAST_SUGGESTION_TIME, last_combined_prompt
    logger.info("📡 [Suggestion Loop] Running...")
    
    consecutive_empty_cycles = 0
    max_empty_cycles = 5  # Reduce frequency if no content
    last_content = ""
    
    while not stop_flag.is_set():
        try:
            # Check priority queue first
            if not critical_queue.empty():
                try:
                    priority, item = critical_queue.get_nowait()
                    # Process critical item immediately
                    suggestion = generate_suggestion(
                        transcript=item["transcript"], 
                        ocr_text=item["ocr_text"], 
                        mode=item.get("mode", "auto"),
                        timeout=5
                    )
                    
                    with suggestion_lock:
                        latest_suggestion = suggestion
                    LAST_SUGGESTION_TIME = time.time()
                    
                    logger.info(f"✅ [CRITICAL SUGGESTION] {suggestion[:100]}{'...' if len(suggestion) > 100 else ''}")
                    continue
                except queue.Empty:
                    pass
            
            # Get latest data
            transcripts = get_latest_transcripts()
            ocr_lines = get_latest_ocr_text()
            ocr_text = " | ".join(ocr_lines)
            
            # Check if we have meaningful content
            has_content = (
                transcripts["interviewee"].strip() or 
                transcripts["interviewer"].strip() or 
                ocr_text.strip()
            )
            
            if not has_content:
                consecutive_empty_cycles += 1
                sleep_time = min(5, consecutive_empty_cycles)  # Increase sleep time when idle
                time.sleep(sleep_time)
                continue
                
            consecutive_empty_cycles = 0  # Reset counter when we have content
            
            # Check if content has changed significantly
            current_content = f"{transcripts['interviewee']}|{transcripts['interviewer']}|{ocr_text}"
            
            # Only generate new suggestion if content has changed significantly
            if similarity(current_content, last_content) > 0.8 and time.time() - LAST_SUGGESTION_TIME < 30:
                time.sleep(suggestion_interval)
                continue
                
            last_content = current_content
            
            # Prepare context for AI
            combined_prompt = f"""
[INTERVIEWEE]: {transcripts['interviewee']}
[INTERVIEWER]: {transcripts['interviewer']}
[SCREEN CONTENT]: {ocr_text}
"""
            
            # Generate suggestion
            suggestion = generate_suggestion(
                transcript=combined_prompt, 
                ocr_text=ocr_text, 
                timeout=5
            )
            
            # Update latest suggestion
            with suggestion_lock:
                latest_suggestion = suggestion
                
            LAST_SUGGESTION_TIME = time.time()
            
            # Log appropriately based on suggestion content
            if suggestion and "No content" not in suggestion and "No suggestions" not in suggestion:
                logger.info(f"✅ [SUGGESTION] {suggestion[:100]}{'...' if len(suggestion) > 100 else ''}")
            else:
                logger.debug("No meaningful suggestion generated")
                
        except Exception as e:
            logger.error(f"Error in suggestion loop: {e}")
            time.sleep(2)  # Brief pause on error
            
        time.sleep(suggestion_interval)
        
    logger.info("🛑 Suggestion loop stopped")