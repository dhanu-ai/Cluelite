# ocr.py
import cv2
import easyocr
import numpy as np
import threading
import time
from PIL import ImageGrab
import logging
from collections import deque
import re
import os
from typing import List, Deque, Tuple

# === CONFIG ===
# Set OCR region to focus on application window, not console
REGION = tuple(map(int, os.getenv("OCR_REGION", "100,100,1500,900").split(','))) if os.getenv("OCR_REGION") else (100, 100, 1500, 900)
INTERVAL = float(os.getenv("OCR_INTERVAL", 3))  # Increased interval
TEXT_LENGTH_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.6
HISTORY_LIMIT = 50  # Reduced history limit
CODE_KEYWORDS = ["def ", "class ", "import ", "function", "return", "print(", "if ", "for ", "while ", "=>", "->", "{", "}", "//", "/*", "*/"]

# === STATE ===
gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
logger = logging.getLogger("Cluelite.OCR")
logger.info(f"Initializing EasyOCR with GPU: {gpu_available}")

try:
    reader = easyocr.Reader(['en'], gpu=gpu_available)
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    raise

seen_lines: Deque[str] = deque(maxlen=HISTORY_LIMIT)
ocr_lock = threading.Lock()
prev_line_count = 0
last_code_detection_time = 0
code_mode = False
processing_enabled = True

def detect_code_pattern(text: str) -> bool:
    """Detect if text contains code patterns."""
    if not text:
        return False
        
    # Check for code keywords
    if any(kw in text for kw in CODE_KEYWORDS):
        return True
        
    # Check for code patterns
    code_patterns = [
        r'\w+\(.*\)',  # Function calls
        r'\w+\s*=\s*\w+',  # Variable assignments
        r'\{.*\}',  # Objects/blocks
        r'\[.*\]',  # Arrays
        r'\.\w+\(',  # Method calls
        r'#.*',  # Python comments
        r'//.*',  # JS comments
        r'/\*.*',  # Multi-line comments
    ]
    
    return any(re.search(p, text) for p in code_patterns)

def preprocess_for_code(img_np: np.ndarray) -> np.ndarray:
    """Enhanced image preprocessing for code detection."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return processed

def preprocess_for_text(img_np: np.ndarray) -> np.ndarray:
    """Image preprocessing for general text detection."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return processed

def grab_screen(region=None) -> np.ndarray:
    """Capture and preprocess screen region."""
    try:
        img = ImageGrab.grab(bbox=region)
        img_np = np.array(img)
        
        global code_mode
        if code_mode:
            return preprocess_for_code(img_np)
        else:
            return preprocess_for_text(img_np)
            
    except Exception as e:
        logger.error(f"Screen capture error: {e}")
        return np.zeros((100, 100), dtype=np.uint8)

def ocr_loop(stop_flag):
    """Main OCR processing loop with enhanced error handling."""
    global prev_line_count, INTERVAL, last_code_detection_time, code_mode, processing_enabled
    logger.info("🟢 OCR Loop started")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while not stop_flag.is_set():
        if not processing_enabled:
            time.sleep(INTERVAL)
            continue
            
        try:
            frame = grab_screen(REGION)
            if frame is None or frame.size == 0:
                time.sleep(INTERVAL)
                continue
                
            results = reader.readtext(frame, paragraph=True)  # Use paragraph mode for better grouping
            new_lines = []
            code_detected = False
            
            for bbox, text, conf in results:
                text = text.strip()
                if conf >= CONFIDENCE_THRESHOLD and len(text) >= TEXT_LENGTH_THRESHOLD:
                    # Filter out common console/terminal text
                    console_patterns = [
                        r'INFO:', r'WARNING:', r'ERROR:', r'DEBUG:',
                        r'HTTP/', r'POST', r'GET', r'localhost',
                        r'python', r'pip', r'npm', r'node',
                        r'uvicorn', r'fastapi', r'127\.0\.0\.1',
                        r'Cluelite', r'OCR', r'STT', r'Groq'
                    ]
                    
                    if any(re.search(p, text, re.IGNORECASE) for p in console_patterns):
                        continue
                        
                    with ocr_lock:
                        if text not in seen_lines:
                            seen_lines.append(text)
                            new_lines.append(text)
                    if detect_code_pattern(text):
                        code_detected = True
                        
            if new_lines:
                logger.info(f"🖥️ OCR detected: {new_lines}")
                code_mode = code_detected or (time.time() - last_code_detection_time < 30)
                if code_detected:
                    last_code_detection_time = time.time()
                    
            prev_line_count = len(seen_lines)
            consecutive_errors = 0  # Reset error count on success
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"OCR Error (attempt {consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many consecutive OCR errors, pausing for 10 seconds")
                time.sleep(10)
                consecutive_errors = 0
                
        time.sleep(INTERVAL)
        
    logger.info("🛑 OCR Loop stopped")

def get_latest_ocr_text() -> List[str]:
    """Get recent OCR results with thread safety."""
    with ocr_lock:
        return list(seen_lines)[-5:]  # Return last 5 items

def clear_ocr_history():
    """Clear OCR history with thread safety."""
    with ocr_lock:
        seen_lines.clear()
    logger.info("OCR History cleared")

def pause_ocr_processing():
    """Pause OCR processing to reduce load."""
    global processing_enabled
    processing_enabled = False

def resume_ocr_processing():
    """Resume OCR processing."""
    global processing_enabled
    processing_enabled = True