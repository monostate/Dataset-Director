# Roadmap

## Version 1.1.0 (Q4 2025)

### Core Improvements
- [ ] Pydantic v2 support (pending KumoAI SDK update)
- [ ] Batch processing for datasets > 100 rows
- [ ] Async job queue for long-running operations
- [ ] WebSocket support for real-time updates

### API Enhancements
- [ ] GraphQL endpoint as alternative to REST
- [ ] Streaming responses for large datasets
- [ ] Webhook callbacks for async operations
- [ ] Pagination for data retrieval endpoints

### Integration Features
- [ ] Direct import from HuggingFace datasets
- [ ] Export to additional formats (Parquet, Arrow, TFRecord)
- [ ] S3/GCS/Azure Blob storage support
- [ ] Integration with popular ML frameworks

## Version 1.2.0 (Q3 2024)

### Enterprise Features
- [ ] Multi-tenant architecture
- [ ] Organization and team management
- [ ] Role-based access control (RBAC)
- [ ] Audit logging with compliance features

### Authentication & Security
- [ ] JWT authentication support
- [ ] OAuth2/OIDC integration
- [ ] API key rotation and management
- [ ] Request signing for webhooks

### Performance & Scalability
- [ ] Horizontal scaling with message queues
- [ ] Advanced caching strategies
- [ ] Connection pooling optimizations
- [ ] CDN integration for static assets

## Version 2.0.0 (Future)

### Platform Evolution
- [ ] Marketplace for datasets and specifications
- [ ] Custom model integration framework
- [ ] Plugin architecture for extensibility
- [ ] Federated learning support

### Advanced ML Features
- [ ] Active learning workflows
- [ ] Auto-labeling with confidence scores
- [ ] Bias detection and mitigation
- [ ] Data quality scoring algorithms

### Developer Experience
- [ ] Official SDKs (Python, JavaScript, Go)
- [ ] CLI tool for dataset management
- [ ] Terraform modules for infrastructure
- [ ] Kubernetes operators

## Completed

### Version 1.0.0 (Released)
- [x] FastAPI service with full API implementation
- [x] KumoRFM integration with LocalTable/LocalGraph
- [x] HuggingFace Hub export functionality
- [x] Redis session persistence
- [x] Docker deployment support
- [x] Fly.io deployment configuration
- [x] Comprehensive test suite (30+ tests)
- [x] API documentation and OpenAPI schema
- [x] Security headers and rate limiting
- [x] Multi-format data upload support

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Priority areas for contribution:
1. Additional file format support
2. Performance optimizations
3. Test coverage improvements
4. Documentation enhancements
5. Bug fixes and issue resolution

## Known Limitations

Current version limitations that contributors might address:
- Maximum 100 rows per upload (safety limit for MVP)
- Pydantic v1.10.x requirement due to KumoAI SDK
- No support for Linux ARM64 in Docker
- In-memory fallback when Redis unavailable
- Single-region deployment (no geo-distribution)