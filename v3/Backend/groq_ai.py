# groq_ai.py — v3 (cleaner context, stricter mode, safer defaults)
import os
import logging
from groq import Groq
from dotenv import load_dotenv
import re
import traceback
from typing import Literal

load_dotenv()
logger = logging.getLogger("Cluelite.groq_ai")

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY = "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"
if not API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

client = Groq(api_key=API_KEY)

# --- heuristics ---
def _is_dsa_problem(text: str) -> bool:
    triggers = [
        "input", "output", "write", "function", "return", "find the",
        "maximum", "sum", "subarray", "integer", "given an", "problem",
        "python function", "array", "example", "constraints", "algorithm",
        "complexity", "time complexity", "space complexity", "efficient"
    ]
    text = text.lower()
    return sum(1 for kw in triggers if len(kw) > 3 and kw in text) >= 2


_CODE_PATTERNS = [
    r'def\s+\w+\(.*?\):', r'class\s+\w+:', r'import\s+\w+',
    r'function\s+\w+\(.*?\)\s*\{', r'\w+\.\w+\(.*?\)',
    r'if\s*\(.*?\)\s*\{?', r'for\s*\(.*?\)\s*\{?', r'while\s*\(.*?\)\s*\{?',
    r'return\s+', r'//.*', r'/\*.*?\*/', r'#.*'
]

def _has_code(text: str) -> bool:
    return any(re.search(p, text, re.DOTALL) for p in _CODE_PATTERNS)


def _extract_snippets(text: str) -> str:
    patterns = [
        r'(def\s+\w+\(.*?\):\n(?:\s+.+\n)+)',
        r'(class\s+\w+:\n(?:\s+.+\n)+)',
        r'(function\s+\w+\(.*?\)\s*\{[^}]+\})',
        r'(\w+\s*=\s*function\(.*?\)\s*\{[^}]+\})',
        r'(if\s*\(.*?\)\s*\{[^}]+\})',
        r'(for\s*\(.*?\)\s*\{[^}]+\})',
        r'(while\s*\(.*?\)\s*\{[^}]+\})',
    ]
    out = []
    for p in patterns:
        out.extend(re.findall(p, text, re.DOTALL))
    return "\n".join(out) if out else text


def _compact(text: str) -> str:
    # remove super short fragments and duplicate pipes
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\|\s*){2,}', '| ', text)
    return text


def generate_suggestion(
    transcript: str,
    ocr_text: str,
    mode: Literal["auto", "solve_code", "response", "analysis"] = "auto",
    timeout: int = 10
) -> str:
    if not (transcript.strip() or ocr_text.strip()):
        return "No content provided for AI processing."

    try:
        clean_ocr = _extract_snippets(ocr_text)
        clean_transcript = _extract_snippets(transcript)
        clean_ocr = _compact(clean_ocr)
        clean_transcript = _compact(clean_transcript)

        # Auto mode selection
        if mode == "auto":
            if _has_code(clean_ocr) or _has_code(clean_transcript) or _is_dsa_problem(clean_ocr):
                mode = "solve_code"
            else:
                mode = "response"

        if mode == "solve_code":
            has_code = _has_code(clean_ocr) or _has_code(clean_transcript)
            if has_code:
                prompt = f"""
[ROLE] Senior Developer Assistant
[TASK] Identify and fix bugs in the following code.

[CODE]
{clean_ocr}

[USER COMMENT]
{clean_transcript}

[INSTRUCTIONS]
1) Point to the exact bug(s) (line refs if possible).
2) Provide corrected code snippet.
3) One-sentence rationale.
4) ≤ 3 sentences total; if no obvious bug, suggest a concrete improvement.
"""
            else:
                prompt = f"""
[ROLE] Senior Algorithm Expert
[TASK] Provide an optimal Python solution.

[PROBLEM]
{ocr_text or transcript}

[INSTRUCTIONS]
- Concise, clean function with edge cases handled.
- Brief complexity (time/space).
- Keep explanation short; well-commented code.
"""

        elif mode == "response":
            prompt = f"""
[ROLE] Technical Interview Assistant
[TASK] Craft a crisp reply to the last point.

[CONVERSATION]
{clean_transcript}

[SCREEN]
{clean_ocr}

[INSTRUCTIONS]
- 1–2 sentences max.
- Precise technical language.
- Address the latest point directly.
"""

        elif mode == "analysis":
            prompt = f"""
[ROLE] Senior Interview Analyst
[TASK] Summarize performance.

[TRANSCRIPT]
{clean_transcript}

[SCREEN]
{clean_ocr}

[REPORT]
- Key strengths
- Specific improvement areas
- Actionable next steps
- Code quality (if relevant)
- Risk areas
(≤6 bullets, professional tone)
"""
        else:
            return "Invalid mode specified."

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a technical interview assistant. Be concise, accurate, and truly helpful. Favor correctness, efficiency, and clarity."},
                {"role": "user", "content": prompt}
            ],
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.4")),  # slightly lower for quality
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "512")),
            top_p=1,
            stream=False,
            timeout=timeout
        )

        result = (completion.choices[0].message.content or "").strip()
        if not result:
            logger.warning("Groq returned empty response.")
            return "No AI response generated."
        print(result)
        return result

    except Exception as e:
        logger.error(f"Groq API error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return f"AI service temporarily unavailable. Error: {type(e).__name__}"
