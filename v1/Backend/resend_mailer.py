#resend mailer file
import os
import json
import requests
import base64
import re
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = "onboarding@resend.dev"
DEFAULT_EMAIL = os.getenv("DEFAULT_EMAIL", "test@cluelite.ai")

EMAIL_REGEX = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')

def is_valid_email(email):
    return bool(EMAIL_REGEX.match(email)) if email else False

def send_email(subject: str, body: str, attachment: str = None, to_email: str = None):
    if not is_valid_email(to_email):
        to_email = DEFAULT_EMAIL
        if not is_valid_email(to_email):
            print("❌ Invalid email configuration")
            return
    
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "from": FROM_EMAIL,
        "to": [to_email],
        "subject": subject,
        "html": f"<pre>{body}</pre>"
    }

    if attachment:
        try:
            with open(attachment, "rb") as f:
                content = base64.b64encode(f.read()).decode()
            payload["attachments"] = [{
                "filename": os.path.basename(attachment),
                "content": content
            }]
        except Exception as e:
            print(f"❌ Attachment error: {e}")

    try:
        print(f"📨 Sending email to {to_email}")
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print("✅ Email sent successfully.")
        else:
            print(f"❌ Email failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception while sending email: {e}")