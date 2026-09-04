import os
import random
import time
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, Tuple

from rag.database import db
from rag.config import get_smtp_config

OTP_EXPIRY_SECONDS = 600  # 10 minutes
SESSION_EXPIRY_SECONDS = 7 * 86400  # 7 days

def generate_otp_code() -> str:
    """
    Generates a 6-digit numeric OTP string.
    """
    return f"{random.randint(100000, 999999)}"

def send_otp_via_smtp(email: str, code: str) -> Tuple[bool, Optional[str]]:
    """
    Sends the OTP verification code via SMTP email using dynamic .env configuration.
    If SMTP is unconfigured or fails, returns (False, error_msg).
    """
    cfg = get_smtp_config()
    host = cfg["host"]
    port = cfg["port"]
    user = cfg["user"]
    password = cfg["pass"]
    sender = cfg["from"] or user

    if not host or not user or not password:
        return False, "SMTP is not configured in .env. Please set SMTP_HOST, SMTP_USER, and SMTP_PASS to receive emails."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your VideoRAG Login Code: {code}"
        msg["From"] = sender
        msg["To"] = email

        text_content = (
            f"Hello,\n\n"
            f"Your login verification code for Video Knowledge RAG is: {code}\n\n"
            f"This code will expire in 10 minutes.\n"
            f"If you did not request this code, you can safely ignore this email.\n"
        )
        html_content = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 520px; margin: 0 auto; padding: 28px; background: #090d16; color: #f8fafc; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="text-align: center; margin-bottom: 24px;">
                <span style="font-size: 36px;">🎬</span>
                <h2 style="color: #ffffff; font-size: 22px; margin: 8px 0 4px 0;">VideoRAG Verification</h2>
                <p style="color: #94a3b8; font-size: 14px; margin: 0;">Multi-User Isolated Video Knowledge Platform</p>
            </div>
            <p style="color: #cbd5e1; font-size: 15px; line-height: 1.5;">Enter the following 6-digit verification code to access your video chats and private knowledge bases:</p>
            <div style="background: rgba(30, 41, 59, 0.8); padding: 20px; border-radius: 12px; text-align: center; margin: 24px 0; border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                <span style="font-size: 36px; font-weight: 800; letter-spacing: 8px; color: #38bdf8; font-family: monospace;">{code}</span>
            </div>
            <p style="color: #64748b; font-size: 13px; text-align: center; margin-top: 16px;">⏱ Code expires in <strong>10 minutes</strong>. Do not share this code with anyone.</p>
        </div>
        """

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=12)
            server.login(user, password)
        else:
            server = smtplib.SMTP(host, port, timeout=12)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(user, password)

        server.sendmail(sender, [email], msg.as_string())
        server.quit()
        return True, None
    except smtplib.SMTPAuthenticationError as auth_err:
        return False, f"Authentication failed with SMTP server. For Gmail, use an App Password instead of your normal password: https://myaccount.google.com/apppasswords ({auth_err})"
    except Exception as e:
        return False, f"SMTP Error: {str(e)}"

def request_otp(email: str) -> Dict[str, Any]:
    """
    Creates and sends an OTP to the given email.
    Logs the user email and generated OTP in the backend console.
    The OTP is NEVER returned to the frontend.
    """
    email = email.lower().strip()
    code = generate_otp_code()
    expires_at = time.time() + OTP_EXPIRY_SECONDS

    # Store in database
    db.store_otp(email, code, expires_at)

    # Attempt SMTP dispatch
    smtp_sent, smtp_err = send_otp_via_smtp(email, code)

    # Log to server console with email and OTP
    print("\n" + "=" * 64)
    print(f"[AUTH OTP] User Email   : {email}")
    print(f"[AUTH OTP] Generated OTP : >>>  {code}  <<<")
    if smtp_sent:
        print(f"[AUTH OTP] Email Status : Successfully dispatched to {email} via SMTP.")
    else:
        print(f"[AUTH OTP] Email Status : SMTP notice ({smtp_err}).")
    print("=" * 64 + "\n")

    return {
        "status": "success",
        "email": email,
        "smtp_sent": smtp_sent,
        "message": f"Verification code sent to {email}. Please check your inbox."
    }



def verify_otp_and_login(email: str, code: str) -> Dict[str, Any]:
    """
    Verifies the OTP code, registers/fetches the user, and creates a 7-day session token.
    """
    email = email.lower().strip()
    code = code.strip()

    valid = db.verify_otp_code(email, code)
    if not valid:
        return {"status": "error", "message": "Invalid or expired verification code."}

    # Fetch or create user record
    user = db.get_or_create_user(email)
    user_id = user["id"]

    # Generate session token
    session_token = secrets.token_hex(32)
    expires_at = time.time() + SESSION_EXPIRY_SECONDS
    db.create_session(user_id, session_token, expires_at)

    return {
        "status": "success",
        "session_token": session_token,
        "user": {
            "id": user_id,
            "email": email
        }
    }

def authenticate_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validates a session token from request headers or cookies.
    """
    if not token:
        return None
    session = db.get_session(token)
    if not session:
        return None
    return {
        "id": session["user_id"],
        "email": session["email"]
    }
