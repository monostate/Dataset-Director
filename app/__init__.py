"""
Vibe Data Director - FastAPI microservice for Dataset Director.
Integrates Kumo SDK and HuggingFace Hub for intelligent dataset curation.
"""

__version__ = "0.1.0"
__author__ = "Vibe Data Team"

from .main import app

__all__ = ["app"]
