"""
Security utilities for Vibe Data Director.
Handles authentication, rate limiting, and security middleware.
"""

import hashlib
import logging
import os
import secrets
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

import redis
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()
logger = logging.getLogger(__name__)

# Context variable for request ID
request_id_context: ContextVar[str] = ContextVar("request_id", default="")

# Configuration
API_KEY = os.getenv("API_KEY")
API_KEY_HASH = hashlib.sha256(API_KEY.encode()).hexdigest() if API_KEY else None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL = int(os.getenv("SESSION_TTL", "1800"))  # 30 minutes default
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# Initialize encryption
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    logger.warning("Generated ephemeral encryption key - set ENCRYPTION_KEY env var in production")
fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

# Initialize Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis not available, falling back to in-memory storage: {e}")
    redis_client = None
    REDIS_AVAILABLE = False

# Rate limiter
def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get authenticated user first
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return f"auth:{hashlib.md5(auth.encode()).hexdigest()[:8]}"
    # Fall back to IP address
    return get_remote_address(request)

limiter = Limiter(key_func=get_client_id)

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", secrets.token_urlsafe(8))
        request_id_context.set(request_id)

        # Process request
        response = await call_next(request)

        # Add security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Remove server header
        # Remove Server header for security
        if "Server" in response.headers:
            del response.headers["Server"]

        return response

# Authentication
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials) -> str:
    """Verify API key with timing attack protection."""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured"
        )

    provided_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()

    # Constant-time comparison
    if not secrets.compare_digest(provided_hash, API_KEY_HASH):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    return credentials.credentials

# Session management
class SessionManager:
    """Secure session management with Redis or in-memory fallback."""

    def __init__(self):
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.memory_timestamps: Dict[str, datetime] = {}

    def create_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Create a new session with TTL."""
        try:
            # Make a copy to avoid modifying original
            session_data = data.copy()

            # Remove non-serializable SDK objects before storing
            if "source_tables" in session_data:
                # Store only table names, not SDK objects
                session_data["source_table_names"] = list(session_data.get("source_tables", {}).keys())
                del session_data["source_tables"]

            # Encrypt sensitive data
            if "samples" in session_data:
                session_data["samples_encrypted"] = self._encrypt_data(session_data["samples"])
                session_data["samples"] = []  # Clear unencrypted samples

            if REDIS_AVAILABLE:
                # Store in Redis with TTL
                redis_client.setex(
                    f"session:{session_id}",
                    SESSION_TTL,
                    self._serialize_data(session_data)
                )
            else:
                # Store in memory (can keep SDK objects here)
                self.memory_store[session_id] = data  # Use original data with SDK objects
                self.memory_timestamps[session_id] = datetime.utcnow()
                self._cleanup_expired_memory_sessions()

            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data with automatic expiration."""
        try:
            if REDIS_AVAILABLE:
                data = redis_client.get(f"session:{session_id}")
                if data:
                    # Refresh TTL on access
                    redis_client.expire(f"session:{session_id}", SESSION_TTL)
                    session = self._deserialize_data(data)
                else:
                    return None
            else:
                # Check memory store
                if session_id not in self.memory_store:
                    return None

                # Check expiration
                timestamp = self.memory_timestamps.get(session_id)
                if timestamp and (datetime.utcnow() - timestamp).seconds > SESSION_TTL:
                    del self.memory_store[session_id]
                    del self.memory_timestamps[session_id]
                    return None

                # Refresh timestamp
                self.memory_timestamps[session_id] = datetime.utcnow()
                session = self.memory_store[session_id]

            # Decrypt samples if needed
            if "samples_encrypted" in session and session["samples_encrypted"]:
                session["samples"] = self._decrypt_data(session["samples_encrypted"])

            return session

        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update existing session."""
        existing = self.get_session(session_id)
        if not existing:
            return False

        # Filter out non-serializable SDK objects from updates
        safe_data = data.copy()
        if "source_tables" in safe_data:
            # Only update source_table_names for Redis storage
            existing["source_table_names"] = list(safe_data.get("source_tables", {}).keys())
            if not REDIS_AVAILABLE:
                # Keep SDK objects in memory storage
                existing["source_tables"] = safe_data["source_tables"]
            del safe_data["source_tables"]

        existing.update(safe_data)
        return self.create_session(session_id, existing)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            if REDIS_AVAILABLE:
                redis_client.delete(f"session:{session_id}")
            else:
                self.memory_store.pop(session_id, None)
                self.memory_timestamps.pop(session_id, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    def _encrypt_data(self, data: Any) -> str:
        """Encrypt sensitive data."""
        import json
        json_str = json.dumps(data)
        encrypted = fernet.encrypt(json_str.encode())
        return encrypted.decode()

    def _decrypt_data(self, encrypted: str) -> Any:
        """Decrypt sensitive data."""
        import json
        decrypted = fernet.decrypt(encrypted.encode())
        return json.loads(decrypted.decode())

    def _serialize_data(self, data: Dict[str, Any]) -> str:
        """Serialize data for storage."""
        import json
        return json.dumps(data, default=str)

    def _deserialize_data(self, data: str) -> Dict[str, Any]:
        """Deserialize data from storage."""
        import json
        return json.loads(data)

    def _cleanup_expired_memory_sessions(self):
        """Clean up expired sessions from memory."""
        if len(self.memory_store) > 100:  # Only cleanup if many sessions
            now = datetime.utcnow()
            expired = [
                sid for sid, timestamp in self.memory_timestamps.items()
                if (now - timestamp).seconds > SESSION_TTL
            ]
            for sid in expired:
                self.memory_store.pop(sid, None)
                self.memory_timestamps.pop(sid, None)

# Initialize session manager
session_manager = SessionManager()

# Input sanitization utilities
def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize text input to prevent XSS and injection."""
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove null bytes
    text = text.replace('\x00', '')

    # Basic HTML entity encoding for safety
    replacements = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '&': '&amp;'
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text.strip()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    import re

    # Remove any path components
    filename = os.path.basename(filename)

    # Remove dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename)

    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]

    return f"{name}{ext}"

# Error handling utilities
def safe_error_response(error: Exception, request_id: str = None) -> Dict[str, str]:
    """Create safe error response without exposing sensitive information."""
    error_map = {
        ValueError: "Invalid input provided",
        KeyError: "Required field missing",
        FileNotFoundError: "Resource not found",
        PermissionError: "Access denied",
    }

    # Get generic message for known error types
    message = error_map.get(type(error), "An error occurred processing your request")

    # Log the actual error with request ID
    logger.error(f"Error in request {request_id}: {str(error)}", exc_info=True)

    return {
        "error": message,
        "request_id": request_id or request_id_context.get()
    }

# Rate limit configurations
RATE_LIMITS = {
    "default": "100/minute",
    "session_init": "10/minute",
    "upload": "20/minute",
    "export": "5/minute",
    "auth": "10/minute"
}

def get_rate_limit(endpoint: str) -> str:
    """Get rate limit for specific endpoint."""
    return RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
