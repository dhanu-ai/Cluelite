import os
import logging
from groq import Groq
from dotenv import load_dotenv
import re
import traceback

load_dotenv()
logger = logging.getLogger("Cluelite.groq_ai")

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_dsa_problem(ocr_text: str) -> bool:
    """Detect if the OCR text describes a DSA problem."""
    triggers = [
        "input", "output", "write", "function", "return", "find the", 
        "maximum", "sum", "subarray", "integer", "given an", "problem", 
        "python function", "array", "example", "constraints"
    ]
    ocr_text = ocr_text.lower()
    return any(kw in ocr_text for kw in triggers if len(kw) > 3)

def detect_code_context(text: str) -> bool:
    """Detect if the context involves code."""
    code_keywords = [
        "def ", "class ", "import ", "function", "return", "print(", 
        "if ", "for ", "while ", "=>", "->", "{", "}", "//", "/*", "*/"
    ]
    return any(kw in text for kw in code_keywords)

def extract_code_snippets(text: str) -> str:
    """Extract and clean code snippets from text."""
    code_patterns = [
        r'def\s+\w+\(.*?\):.*?return',  # Python functions
        r'function\s+\w+\(.*?\)\s*{.*?}',  # JS functions
        r'class\s+\w+:\s*def',  # Python classes
        r'\w+\s*=\s*function\(.*?\)\s*{',  # JS function assignment
        r'print\(.*?\)',  # Print statements
        r'if\s*\(.*?\)\s*{',  # Conditionals
        r'for\s*\(.*?\)\s*{',  # Loops
    ]
    snippets = []
    for pattern in code_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        snippets.extend(matches)
    return "\n".join(snippets) if snippets else text

def generate_suggestion(transcript: str, ocr_text: str, mode: str = "auto", timeout: int = 10) -> str:
    """
    Generate smart suggestions based on live transcript and screen context.
    """
    if not transcript.strip() and not ocr_text.strip():
        return "No content provided for AI processing."

    clean_ocr = extract_code_snippets(ocr_text)
    clean_transcript = extract_code_snippets(transcript)

    # Mode auto-detection
    if mode == "auto":
        if detect_code_context(clean_ocr) or detect_code_context(clean_transcript):
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
1. Identify the exact bug(s)
2. Provide the corrected code snippet
3. Explain the fix in one sentence
4. Keep response concise (max 3 sentences)
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
"""

    elif mode == "response":
        prompt = f"""
[ROLE] Interview Assistant
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
"""

    elif mode == "analysis":
        prompt = f"""
[ROLE] Interview Analyst
[TASK] Provide performance summary:

[TRANSCRIPT]
{clean_transcript}

[SCREEN CONTENT]
{clean_ocr}

[REPORT REQUIREMENTS]
1. Strengths (technical and communication)
2. Weaknesses (specific technical gaps)
3. Improvement Tips (actionable)
4. Code quality assessment (if applicable)
5. Risk areas

[FORMAT]
- Bullet points
- Max 6 bullets
- Include code fixes if relevant
"""
    else:
        logger.warning(f"Invalid mode passed: {mode}")
        return "Invalid mode"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are the interviewee, attend the interview and ace it. If it's 'coding round' do not hesitate to code. Your goal is to ace the interview so, there is no need to hesitate."},
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
            return "[No response from AI]"

        return result

    except Exception as e:
        logger.error(f"Groq API error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return f"[AI service unavailable: {type(e).__name__} - {e}]"
