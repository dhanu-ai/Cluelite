# ocr.py — v3 (event-driven, change-only OCR with better preprocessing & early start)
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
from typing import List, Deque

# === CONFIG ===
REGION = tuple(map(int, os.getenv("OCR_REGION", "400,100,1600,800").split(','))) if os.getenv("OCR_REGION") else (300, 150, 1300, 700)
INTERVAL = float(os.getenv("OCR_INTERVAL", "0.8"))  # Faster polling so we can edge-trigger quickly
TEXT_LENGTH_THRESHOLD = int(os.getenv("OCR_TEXT_LENGTH_THRESHOLD", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
HISTORY_LIMIT = int(os.getenv("OCR_HISTORY_LIMIT", "120"))
CODE_KEYWORDS = [
    "def ", "class ", "import ", "function", "return", "print(", "if ", "for ", "while ",
    "=>", "->", "{", "}", "//", "/*", "*/", "from ", "const ", "let "
]
CHANGE_MIN_DELTA = int(os.getenv("OCR_CHANGE_MIN_DELTA", "1"))  # how many new lines to count as change
MAX_LINES_RETURNED = int(os.getenv("OCR_MAX_LINES_RETURNED", "16"))

# === STATE ===
gpu_available = False
try:
    gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    gpu_available = False

logger = logging.getLogger("Cluelite.OCR")
logger.info(f"Initializing EasyOCR with GPU: {gpu_available}")

try:
    reader = easyocr.Reader(['en'], gpu=gpu_available)
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    raise

seen_lines: Deque[str] = deque(maxlen=HISTORY_LIMIT)
ocr_lock = threading.Lock()

# Change-notification primitives
_ocr_change_event = threading.Event()
_ocr_revision = 0  # monotonically increasing on change

# Code-mode hints
last_code_detection_time = 0.0
code_mode = False
CODE_MODE_HOLD_SEC = 30.0


def detect_code_pattern(text: str) -> bool:
    if not text:
        return False
    if any(kw in text for kw in CODE_KEYWORDS):
        return True
    code_patterns = [
        r'\w+\(.*\)', r'\w+\s*=\s*\w+', r'\{.*\}', r'\[.*\]',
        r'\.\w+\(', r'#.*', r'//.*', r'/\*.*', r'from\s+\w+\s+import',
        r'const\s+\w+\s*=', r'let\s+\w+\s*='
    ]
    return any(re.search(p, text) for p in code_patterns)


def preprocess_for_code(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    processed = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)
    return processed


def preprocess_for_text(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    processed = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return processed


def grab_screen(region=None) -> np.ndarray:
    try:
        img = ImageGrab.grab(bbox=region)
        img_np = np.array(img)
        if code_mode:
            return preprocess_for_code(img_np)
        return preprocess_for_text(img_np)
    except Exception as e:
        logger.error(f"Screen capture error: {e}")
        return np.zeros((100, 100), dtype=np.uint8)


def _bump_revision():
    global _ocr_revision
    _ocr_revision += 1
    _ocr_change_event.set()


def ocr_loop(stop_flag: threading.Event):
    """Continuously OCR the screen, but only signal changes when *new* lines appear."""
    global last_code_detection_time, code_mode
    logger.info("🟢 OCR Loop started (v3)")
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not stop_flag.is_set():
        try:
            frame = grab_screen(REGION)
            if frame is None or frame.size == 0:
                time.sleep(INTERVAL)
                continue

            results = reader.readtext(frame)
            new_lines = []
            code_detected = False

            for _, text, conf in results:
                text = (text or "").strip()
                if conf >= CONFIDENCE_THRESHOLD and len(text) >= TEXT_LENGTH_THRESHOLD:
                    if detect_code_pattern(text):
                        code_detected = True
                    with ocr_lock:
                        if text not in seen_lines:
                            seen_lines.append(text)
                            new_lines.append(text)

            # maintain code mode sticky
            now = time.time()
            if code_detected:
                last_code_detection_time = now
            code_mode = code_detected or (now - last_code_detection_time) < CODE_MODE_HOLD_SEC

            if len(new_lines) >= CHANGE_MIN_DELTA:
                logger.info(f"🖥️ OCR change: {new_lines}")
                _bump_revision()

            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"OCR Error (attempt {consecutive_errors}): {e}")
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many OCR errors; backing off 10s")
                time.sleep(10)
                consecutive_errors = 0
        time.sleep(INTERVAL)

    logger.info("🛑 OCR Loop stopped")


def get_latest_ocr_text() -> List[str]:
    # Return only the last N lines for compact context
    with ocr_lock:
        return list(seen_lines)[-MAX_LINES_RETURNED:]


def clear_ocr_history():
    with ocr_lock:
        seen_lines.clear()
    logger.info("OCR History cleared")
    _bump_revision()


# === Change-driven API for suggestion_loop ===
def get_ocr_revision() -> int:
    return _ocr_revision


def wait_for_ocr_change(since_rev: int, timeout: float = 0.5) -> bool:
    """
    Block until OCR revision advances beyond since_rev or timeout expires.
    Returns True if a change occurred.
    """
    if _ocr_revision > since_rev:
        return True
    _ocr_change_event.wait(timeout)
    _ocr_change_event.clear()
    return _ocr_revision > since_rev
