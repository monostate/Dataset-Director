# Contributing to Dataset Director

First off, thank you for considering contributing to Dataset Director! It's people like you that make Dataset Director such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed and expected**
* **Include logs and error messages**
* **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a detailed description of the suggested enhancement**
* **Provide specific examples to demonstrate the enhancement**
* **Describe the current behavior and expected behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing style
6. Issue that pull request!

## Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/dataset-director.git
cd dataset-director
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Set up pre-commit hooks (optional but recommended)**
```bash
pre-commit install
```

5. **Run tests**
```bash
pytest tests/ -v
```

## Style Guidelines

### Python Style

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use type hints where appropriate
* Write docstrings for all functions and classes
* Keep line length to 88 characters (Black's default)

### Code Formatting

We use the following tools to maintain code quality:
* **Black** for code formatting
* **Ruff** for linting
* **MyPy** for type checking

Run formatters and linters:
```bash
black app/ tests/
ruff check app/ tests/
mypy app/
```

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
Add HuggingFace dataset validation

- Validate dataset format before export
- Add size limits for free tier users
- Include progress callbacks for large uploads

Fixes #123
```

## Testing

* Write tests for any new functionality
* Ensure all tests pass before submitting PR
* Aim for high test coverage (>80%)
* Use pytest fixtures for reusable test components

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_basic.py

# Run with verbose output
pytest -v
```

## Documentation

* Update the README.md if needed
* Update API.md for endpoint changes
* Add docstrings to new functions/classes
* Include type hints
* Add examples for complex functionality

## Project Structure

```
dataset-director/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application
│   ├── kumo_client.py     # Kumo SDK integration
│   ├── hf_export.py       # HuggingFace export
│   └── security.py        # Security middleware
├── tests/                  # Test files
├── docs/                   # Additional documentation
├── scripts/               # Utility scripts
└── deployment/            # Deployment configurations
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a tagged release on GitHub
4. Deploy to production

## Questions?

Feel free to open an issue for any questions about contributing!

## Recognition

Contributors will be recognized in our README.md file. Thank you for your contributions!
