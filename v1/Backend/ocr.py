#ocr file
import cv2
import easyocr
import numpy as np
import threading
import time
import torch
from PIL import ImageGrab
import logging
from collections import deque
import re
import os

# === CONFIG ===
REGION = tuple(map(int, os.getenv("OCR_REGION", "400,100,1600,800").split(','))) \
    if os.getenv("OCR_REGION") else (400, 100, 1600, 800)
INTERVAL = float(os.getenv("OCR_INTERVAL", 2))  # seconds
TEXT_LENGTH_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.6
HISTORY_LIMIT = 100
CODE_KEYWORDS = ["def ", "class ", "import ", "function", "return", "print(", "if ", "for ", "while ", "=>", "->", "{", "}", "//", "/*", "*/"]

# === STATE ===
gpu_available = torch.cuda.is_available()
logger = logging.getLogger("Cluelite.OCR")
logger.info(f"Initializing EasyOCR with GPU: {gpu_available}")
reader = easyocr.Reader(['en'], gpu=gpu_available)
seen_lines = deque(maxlen=HISTORY_LIMIT)
ocr_lock = threading.Lock()
prev_line_count = 0
last_code_detection_time = 0
code_mode = False

def detect_code_pattern(text):
    """Heuristic to detect code-like patterns"""
    if not text:
        return False
    
    # Check for code-specific keywords
    if any(kw in text for kw in CODE_KEYWORDS):
        return True
        
    # Check for code-like patterns (indentation, operators, etc.)
    code_patterns = [
        r'\w+\(.*\)',    # Function calls
        r'\w+\s*=\s*\w+',  # Assignments
        r'\{.*\}',       # Curly braces
        r'\[.*\]',       # Square brackets
        r'\.\w+\(',      # Method calls
        r'#.*',          # Comments
        r'//.*',         # Comments
        r'/\*.*'         # Comments
    ]
    
    return any(re.search(pattern, text) for pattern in code_patterns)

def preprocess_for_code(img_np):
    """Special preprocessing for code screenshots"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Use global thresholding for better monospace font recognition
        _, processed = cv2.threshold(equalized, 150, 255, cv2.THRESH_BINARY)
        
        return processed
    except Exception as e:
        logger.error(f"Code preprocessing error: {e}")
        return img_np

def grab_screen(region=None):
    try:
        img = ImageGrab.grab(bbox=region)
        img_np = np.array(img)
        
        # Use code-specific preprocessing if in code mode
        global code_mode
        if code_mode:
            return preprocess_for_code(img_np)
        
        # Standard preprocessing for other content
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        return processed
    except Exception as e:
        logger.error(f"Screen Grab Error: {e}")
        return None

def ocr_loop(stop_flag):
    global prev_line_count, INTERVAL, last_code_detection_time, code_mode
    logger.info("🟢 OCR Loop started")
    
    while not stop_flag.is_set():
        frame = grab_screen(REGION)
        if frame is None:
            time.sleep(INTERVAL)
            continue

        try:
            results = reader.readtext(frame)
            new_lines = []
            code_content = False

            for _, text, conf in results:
                text = text.strip()
                if conf >= CONFIDENCE_THRESHOLD and len(text) >= TEXT_LENGTH_THRESHOLD:
                    with ocr_lock:
                        if text not in seen_lines:
                            seen_lines.append(text)
                            new_lines.append(text)
                            
                    # Check if this looks like code
                    if detect_code_pattern(text):
                        code_content = True

            if new_lines:
                logger.info(f"🖥️ OCR New: {new_lines}")
                
                # Update code mode status
                if code_content:
                    code_mode = True
                    last_code_detection_time = time.time()
                elif time.time() - last_code_detection_time > 30:  # 30s since last code
                    code_mode = False
            else:
                logger.debug("🖥️ OCR: No new lines")

            # Adaptive interval logic
            current_line_count = len(seen_lines)
            if abs(current_line_count - prev_line_count) < 2:
                INTERVAL = min(INTERVAL * 1.5, 10)
            else:
                INTERVAL = max(float(os.getenv("OCR_INTERVAL", 2)), 1)
            prev_line_count = current_line_count

        except Exception as e:
            logger.error(f"OCR Processing Error: {e}")

        time.sleep(INTERVAL)
    logger.info("🛑 OCR Loop stopped")

def get_latest_ocr_text():
    with ocr_lock:
        return list(seen_lines)[-5:]

def clear_ocr_history():
    global seen_lines
    with ocr_lock:
        seen_lines.clear()
    logger.info("OCR History cleared")