# Changelog

All notable changes to Dataset Director will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-18

### Added
- Initial release of Dataset Director
- FastAPI-based REST API with automatic OpenAPI documentation
- KumoRFM integration for intelligent dataset predictions
- HuggingFace Hub export functionality
- Session management with Redis support
- Multi-format data upload (CSV, JSON, multipart)
- Bearer token authentication
- Rate limiting per endpoint
- Security headers middleware
- Docker support with multi-stage builds
- Fly.io deployment configuration
- Comprehensive test suite (30+ tests)
- Complete API documentation

### Features
- `/session/init` - Initialize dataset curation sessions
- `/session/seed_upload` - Upload seed data via files
- `/session/seed_upload_json` - Upload seed data via JSON
- `/plan/coverage` - Get coverage predictions per class
- `/plan/specs` - Get recommended next specifications
- `/export/hf` - Export curated dataset to HuggingFace

### Known Issues
- Requires Pydantic v1.10.x due to KumoAI SDK compatibility
- KumoRFM does not support Linux ARM64 architecture
- Different from Kumo Enterprise (uses LocalTable/LocalGraph instead of cloud uploads)

### Security
- API key authentication required for all endpoints except health
- Input sanitization and validation
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- Rate limiting to prevent abuse

### Dependencies
- FastAPI 0.115.0
- Pydantic 1.10.22
- KumoAI 2.6.0
- HuggingFace Hub 0.20.0+
- Redis 5.0.1 (optional)

## [Unreleased]

### To Do
- Add support for Pydantic v2 when KumoAI SDK updates
- Implement batch processing for large datasets
- Add webhook support for async operations
- Support for additional file formats
- Enhanced monitoring and metrics
- WebSocket support for real-time updates
