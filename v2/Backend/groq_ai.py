# groq_ai.py
import os
import logging
from groq import Groq
from dotenv import load_dotenv
import re
import traceback
from typing import Optional, Literal
import hashlib
from ratelimiter import RateLimiter
import requests
from requests.adapters import HTTPAdapter

load_dotenv()
logger = logging.getLogger("Cluelite.groq_ai")

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

# Create a session with connection pooling
session = requests.Session()
session.mount('https://', HTTPAdapter(pool_connections=10, pool_maxsize=100))
client = Groq(api_key=API_KEY, http_client=session)

# Response cache
response_cache = {}
CACHE_SIZE_LIMIT = 1000  # Keep only the last 1000 requests

def get_request_hash(prompt):
    """Generate a hash for the prompt to use as cache key."""
    return hashlib.md5(prompt.encode()).hexdigest()

# Rate limiter: 30 calls per minute
@RateLimiter(max_calls=30, period=60)
def rate_limited_generate_suggestion():
    """Placeholder for rate-limited function."""
    pass

def is_dsa_problem(ocr_text: str) -> bool:
    """Detect if the OCR text describes a DSA problem with improved accuracy."""
    triggers = [
        "input", "output", "write", "function", "return", "find the", 
        "maximum", "sum", "subarray", "integer", "given an", "problem", 
        "python function", "array", "example", "constraints", "algorithm",
        "complexity", "time complexity", "space complexity", "efficient"
    ]
    
    ocr_text = ocr_text.lower()
    trigger_count = sum(1 for kw in triggers if len(kw) > 3 and kw in ocr_text)
    return trigger_count >= 2  # Require multiple triggers for better accuracy

def detect_code_context(text: str) -> bool:
    """Detect if the context involves code with improved patterns."""
    code_patterns = [
        r'def\s+\w+\(.*?\):',  # Python functions
        r'class\s+\w+:',  # Classes
        r'import\s+\w+',  # Imports
        r'function\s+\w+\(.*?\)\s*\{',  # JS functions
        r'\w+\.\w+\(.*?\)',  # Method calls
        r'if\s*\(.*?\)\s*\{?',  # Conditionals
        r'for\s*\(.*?\)\s*\{?',  # Loops
        r'while\s*\(.*?\)\s*\{?',  # While loops
        r'return\s+',  # Return statements
        r'//.*',  # Single line comments
        r'/\*.*?\*/',  # Multi-line comments
        r'#.*',  # Python comments
    ]
    
    return any(re.search(pattern, text, re.DOTALL) for pattern in code_patterns)

def enhanced_ocr_cleanup(text: str) -> str:
    """Clean up OCR text by removing common artifacts."""
    if not text:
        return text
        
    # Remove common OCR artifacts
    artifacts = [
        r'\|\|+', r'\.\.+', r'__+',  # Repeated characters
        r'[^\w\s\(\)\{\}\[\]\.\,\;\+\-\*\/=]'  # Strange symbols
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_valid_content(text: str) -> bool:
    """Check if text has meaningful content."""
    if not text:
        return False
        
    # Check if text has at least 3 alphanumeric characters
    return len(re.findall(r'[a-zA-Z0-9]', text)) >= 3

def extract_code_snippets(text: str) -> str:
    """Extract and clean code snippets from text with improved accuracy."""
    code_patterns = [
        r'(def\s+\w+\(.*?\):\n(?:\s+.+\n)+)',  # Python functions with body
        r'(class\s+\w+:\n(?:\s+.+\n)+)',  # Python classes with body
        r'(function\s+\w+\(.*?\)\s*\{[^}]+\})',  # JS functions with body
        r'(\w+\s*=\s*function\(.*?\)\s*\{[^}]+\})',  # JS function assignment
        r'(if\s*\(.*?\)\s*\{[^}]+\})',  # If statements
        r'(for\s*\(.*?\)\s*\{[^}]+\})',  # For loops
        r'(while\s*\(.*?\)\s*\{[^}]+\})',  # While loops
    ]
    
    snippets = []
    for pattern in code_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        snippets.extend(matches)
    
    return "\n".join(snippets) if snippets else text

def generate_suggestion(
    transcript: str, 
    ocr_text: str, 
    mode: Literal["auto", "solve_code", "response", "analysis"] = "auto", 
    timeout: int = 10
) -> str:
    """
    Generate smart suggestions based on live transcript and screen context.
    """
    if not transcript.strip() and not ocr_text.strip():
        return "No content provided for AI processing."

    try:
        # Clean up OCR text
        clean_ocr = enhanced_ocr_cleanup(extract_code_snippets(ocr_text))
        clean_transcript = enhanced_ocr_cleanup(extract_code_snippets(transcript))
        
        # Skip if no valid content
        if not is_valid_content(clean_ocr) and not is_valid_content(clean_transcript):
            return "No meaningful content detected for processing."

        # Mode auto-detection with improved logic
        if mode == "auto":
            code_in_ocr = detect_code_context(clean_ocr)
            code_in_transcript = detect_code_context(clean_transcript)
            
            if code_in_ocr or code_in_transcript:
                mode = "solve_code"
            elif is_dsa_problem(clean_ocr):
                mode = "solve_code"
            else:
                mode = "response"

        logger.info(f"[Groq AI] Mode selected: {mode}")

        # Solve code mode: bug fix OR full solution
        if mode == "solve_code":
            has_code = detect_code_context(clean_ocr) or detect_code_context(clean_transcript)
            logger.info(f"[Groq AI] Detected code in input: {has_code}")

            if has_code:
                prompt = f"""
[ROLE] Senior Developer Assistant
[TASK] Identify and fix bugs in the following code

[CODE]
{clean_ocr}

[USER COMMENT]
{clean_transcript}

[INSTRUCTIONS]
1. Identify the exact bug(s) with line numbers if possible
2. Provide the corrected code snippet
3. Explain the fix in one sentence
4. Keep response concise (max 3 sentences)
5. If no bugs found, suggest improvements
"""
            else:
                prompt = f"""
[ROLE] Senior Algorithm Expert
[TASK] Write a complete, optimal Python solution to the following problem.

[PROBLEM STATEMENT]
{ocr_text or transcript}

[INSTRUCTIONS]
1. Analyze the problem requirements and constraints.
2. Provide a clean, efficient Python function solving it.
3. Include brief complexity analysis (time and space).
4. Keep explanation short, code well-commented.
5. Handle edge cases appropriately.
"""

        elif mode == "response":
            prompt = f"""
[ROLE] Technical Interview Assistant
[TASK] Craft a technical response based on:

[CONVERSATION]
{clean_transcript}

[SCREEN CONTENT]
{clean_ocr}

[INSTRUCTIONS]
- Respond in 1-2 sentences maximum
- Use technical terms precisely
- Maintain professional tone
- Address the most recent point
- Be concise and directly helpful
"""

        elif mode == "analysis":
            prompt = f"""
[ROLE] Senior Interview Analyst
[TASK] Provide performance summary:

[TRANSCRIPT]
{clean_transcript}

[SCREEN CONTEXT]
{clean_ocr}

[REPORT REQUIREMENTS]
1. Key strengths (technical and communication)
2. Specific areas for improvement
3. Actionable recommendations
4. Code quality assessment (if applicable)
5. Risk areas to address

[FORMAT]
- Bullet points
- Max 6 bullets
- Professional tone
- Focus on measurable improvements
"""
        else:
            logger.warning(f"Invalid mode passed: {mode}")
            return "Invalid mode specified"

        # Check cache first
        request_hash = get_request_hash(prompt)
        if request_hash in response_cache:
            logger.info("Returning cached response")
            return response_cache[request_hash]

        # Apply rate limiting
        rate_limited_generate_suggestion()

        # Make API call with enhanced error handling
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a technical interview assistant. Provide concise, accurate, and helpful responses. For coding problems, focus on correctness, efficiency, and clarity."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            stream=False,
            timeout=timeout
        )

        result = completion.choices[0].message.content.strip()
        if not result:
            logger.warning("Groq API returned empty response.")
            return "No AI response generated. Please try again."

        # Cache the result
        if len(response_cache) >= CACHE_SIZE_LIMIT:
            # Remove the oldest item if cache is full
            response_cache.pop(next(iter(response_cache)))
        response_cache[request_hash] = result

        return result

    except Exception as e:
        logger.error(f"Groq API error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return f"AI service temporarily unavailable. Please try again shortly. Error: {type(e).__name__}"