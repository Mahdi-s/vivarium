# Contributing to Vivarium

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone <your-fork-url>
   cd abstractAgentMachine
   ```

2. **Install development dependencies:**
   ```bash
   uv sync --extra cognitive --extra interpretability
   ```

3. **Set up pre-commit hooks (optional):**
   ```bash
   # Install pre-commit if desired
   pip install pre-commit
   pre-commit install
   ```

## Code Style

- **Type Hints**: All functions should have type hints
- **Docstrings**: Public functions and classes should have docstrings
- **Formatting**: Follow PEP 8 style guidelines
- **Imports**: Use absolute imports from `aam.*`

## Testing

Before submitting a PR:

1. **Run Phase 1 smoke test:**
   ```bash
   PYTHONPATH=src vvm phase1 --steps 10 --agents 2 --seed 42 --db test.db
   ```

2. **Verify determinism:**
   ```bash
   PYTHONPATH=src vvm phase1 --steps 10 --agents 2 --seed 42 --db test1.db
   PYTHONPATH=src vvm phase1 --steps 10 --agents 2 --seed 42 --db test2.db
   # Compare databases
   ```

3. **Check for linter errors:**
   ```bash
   python -m py_compile src/aam/*.py
   ```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Update documentation if needed
4. Run tests locally
5. Submit a PR with a clear description of changes

## Architecture Principles

When contributing, please respect:

- **Separation of Concerns**: Platform (WorldEngine) vs Agent (Policy) vs Channel
- **Determinism**: All randomness must be seeded and reproducible
- **Trace as Truth**: Database state is derivative of trace events
- **Extensibility**: Use protocols/interfaces for pluggable components

## Questions?

Open an issue for discussion before making large changes.

