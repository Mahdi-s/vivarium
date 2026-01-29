# Setup and Environment Guide

This document provides detailed instructions for setting up and running Vivarium across different phases and environments.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Environment Configuration](#environment-configuration)
- [Phase-Specific Setup](#phase-specific-setup)
- [Model Setup](#model-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows)
- **RAM**: 4GB minimum (8GB+ recommended for Phase 3 with local models)
- **Disk Space**: 
  - Base installation: ~100MB
  - Phase 2 (cognitive): +500MB
  - Phase 3 (interpretability): +2GB (PyTorch + TransformerLens)
  - Models: Varies (GGUF models typically 1-20GB each)

### Recommended Requirements

- **Python**: 3.12
- **RAM**: 16GB+ (for running local models)
- **GPU**: Optional but recommended for Phase 3 (CUDA-compatible for PyTorch)
- **Disk Space**: 50GB+ free (for models and activation shards)

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or: pip install uv
   ```

2. **Clone and install:**
   ```bash
   git clone <repository-url>
   cd abstractAgentMachine
   
   # Phase 1 only (minimal)
   uv sync
   
   # Phase 2 (cognitive layer)
   uv sync --extra cognitive
   
   # Phase 3 (full with interpretability)
   uv sync --extra cognitive --extra interpretability
   ```

3. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or: .venv\Scripts\activate  # Windows
   ```

### Method 2: Using pip

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd abstractAgentMachine
   ```

2. **Create virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   # Phase 1 only
   pip install -e .
   
   # Phase 2
   pip install -e .[cognitive]
   
   # Phase 3 (full)
   pip install -e .[cognitive,interpretability]
   ```

### Method 3: Development Installation

For active development:

```bash
git clone <repository-url>
cd abstractAgentMachine
uv sync --extra cognitive --extra interpretability --dev
# or: pip install -e ".[cognitive,interpretability]"
```

## Environment Configuration

### Python Path Setup

Since the project uses a `src/` layout, you have two options:

**Option A: Set PYTHONPATH (Quick)**
```bash
export PYTHONPATH=/path/to/abstractAgentMachine/src
vvm phase1 --steps 10
```

**Option B: Install Package (Recommended)**
```bash
uv sync  # or: pip install -e .
vvm phase1 --steps 10
```

### Environment Variables

#### For External LLM Providers (Phase 2)

**OpenAI:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Anthropic:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Azure OpenAI:**
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
```

#### For HuggingFace Models (Phase 3)

**HuggingFace Token (for gated models):**
```bash
export HF_TOKEN="hf_..."
# Or login: huggingface-cli login
```

**HuggingFace Cache Directory:**
```bash
export HF_HOME="/path/to/cache"  # Default: ~/.cache/huggingface
```

#### For Local llama.cpp Server

No environment variables needed. The server runs on `http://127.0.0.1:8081/v1` by default.

**macOS Apple Silicon:** The server automatically uses Metal GPU acceleration by default (all layers on GPU). This provides significant performance improvements. You can verify Metal is working by checking server logs for "ggml_metal_device_init: GPU name: Apple M*".

#### AAM Path Configuration

The AAM framework uses centralized settings (`src/aam/settings.py`) that can be customized via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AAM_MODEL_DIR` | Directory for model weights (GGUF, Safetensors) | `<PROJECT_ROOT>/models` |
| `AAM_LLAMA_CPP_ROOT` | Path to llama.cpp installation | `<PROJECT_ROOT>/third_party/llama.cpp` |
| `AAM_ARTIFACTS_DIR` | Root directory for simulation outputs | `<PROJECT_ROOT>/runs` |
| `AAM_HF_CACHE` | HuggingFace cache directory | `<AAM_MODEL_DIR>/huggingface_cache` |

**Example `.env` file:**

Create a `.env` file in the repository root to set these variables. Install `python-dotenv` (included in `[dev]` extras) to auto-load them:

```bash
# .env
AAM_MODEL_DIR=/path/to/large/storage/models
AAM_LLAMA_CPP_ROOT=/usr/local/llama.cpp
AAM_ARTIFACTS_DIR=/data/aam/runs
AAM_HF_CACHE=/path/to/hf/cache
```

**Usage in Python:**

```python
from aam.settings import settings

print(settings.MODEL_DIR)       # Path to model directory
print(settings.LLAMA_CPP_ROOT)  # Path to llama.cpp
print(settings.ARTIFACTS_DIR)   # Path to run outputs
print(settings.HF_CACHE)        # HuggingFace cache path

# Validate paths exist
status = settings.validate_paths()
print(status)  # {'MODEL_DIR': True, 'LLAMA_CPP_ROOT': False, ...}
```

### CUDA Setup (Optional, for Phase 3 GPU acceleration)

If you have an NVIDIA GPU:

1. **Install CUDA toolkit** (version 11.8 or 12.1 recommended)

2. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU access:**
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Phase-Specific Setup

### Phase 1: Core Simulation

**Dependencies:**
- `pydantic>=2.6` (automatic)

**Setup:**
```bash
uv sync
# or: pip install -e .
```

**Test:**
```bash
PYTHONPATH=src vvm phase1 --steps 10 --agents 2 --seed 42 --db test.db
```

### Phase 2: Cognitive Layer

**Dependencies:**
- Phase 1 dependencies
- `langgraph>=0.2.0`
- `litellm>=1.40.0`
- `json-repair` (for text fallback parsing)

**Setup:**
```bash
uv sync --extra cognitive
# or: pip install -e .[cognitive]
```

**Additional packages (if needed):**
```bash
pip install json-repair  # For dirty JSON parsing
```

**Test:**
```bash
PYTHONPATH=src vvm phase2 --steps 5 --agents 2 --mock-llm --db test_phase2.db
```

### Phase 3: Interpretability Layer

**Dependencies:**
- Phase 2 dependencies
- `torch>=2.1.0`
- `transformer-lens>=1.14.0`
- `safetensors>=0.4.0`
- `transformers` (for HuggingFace model loading)
- `accelerate` (for model loading optimization)

**Setup:**
```bash
uv sync --extra cognitive --extra interpretability
# or: pip install -e .[cognitive,interpretability]
```

**Additional packages:**
```bash
pip install transformers accelerate  # Usually installed with transformer-lens
```

**Test:**
```bash
# List hooks (requires model download)
PYTHONPATH=src vvm phase3 \
  --model-id gpt2 \
  --list-hooks
```

**Note:** First run will download the model from HuggingFace (can be several GB).

## Model Setup

### GGUF Models (for llama.cpp / Phase 2)

**Option 1: Use Ollama models**

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama3.2:1b`
3. Models are stored in `~/.ollama/models/`
4. Use `vvm llama list` to discover them

**Option 2: Use LM Studio models**

1. Install [LM Studio](https://lmstudio.ai/)
2. Download models through the UI
3. Models are stored in `~/Library/Application Support/LM Studio/models/` (macOS)
4. Use `vvm llama list` to discover them

**Option 3: Manual placement**

1. Download GGUF files manually
2. Place in `models/` directory:
   ```bash
   mkdir -p models
   cp /path/to/model.gguf models/
   ```

**Export discovered models:**
```bash
PYTHONPATH=src vvm llama export
# Creates symlinks in models/ directory
```

### HuggingFace Models (for Phase 3)

**Public models:**
```bash
# No setup needed, specify model ID:
PYTHONPATH=src vvm phase3 --model-id gpt2 --list-hooks
```

**Gated models (require authentication):**
```bash
huggingface-cli login
# Enter your HF token
```

**Local models:**
If you have models downloaded locally:
```bash
# Use local path as model-id
PYTHONPATH=src vvm phase3 \
  --model-id /path/to/local/model \
  --list-hooks
```

## Verification

### Quick Health Check

Run this sequence to verify all phases:

```bash
# Phase 1
PYTHONPATH=src vvm phase1 --steps 3 --agents 2 --seed 42 --db verify_phase1.db
echo "Phase 1: ✓"

# Phase 2 (mock)
PYTHONPATH=src vvm phase2 --steps 3 --agents 2 --seed 42 --mock-llm --db verify_phase2.db
echo "Phase 2: ✓"

# Phase 3 (if installed, requires model)
# PYTHONPATH=src vvm phase3 --model-id gpt2 --list-hooks
# echo "Phase 3: ✓"
```

### Database Validation

Check that databases are created correctly:

```bash
python3 << EOF
import sqlite3
import sys

db = sys.argv[1]
conn = sqlite3.connect(db)
print(f"Runs: {conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]}")
print(f"Trace rows: {conn.execute('SELECT COUNT(*) FROM trace').fetchone()[0]}")
print(f"Messages: {conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]}")
conn.close()
EOF verify_phase1.db
```

### Determinism Test

Run the same simulation twice and compare:

```bash
PYTHONPATH=src vvm phase1 --steps 10 --agents 2 --seed 42 --db test1.db
PYTHONPATH=src vvm phase1 --steps 10 --agents 2 --seed 42 --db test2.db

# Compare databases (should be identical)
python3 << EOF
import sqlite3
db1 = sqlite3.connect('test1.db')
db2 = sqlite3.connect('test2.db')
count1 = db1.execute('SELECT COUNT(*) FROM trace').fetchone()[0]
count2 = db2.execute('SELECT COUNT(*) FROM trace').fetchone()[0]
print(f"DB1 trace rows: {count1}")
print(f"DB2 trace rows: {count2}")
assert count1 == count2, "Databases differ!"
print("✓ Determinism verified")
EOF
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'aam'**

**Solution:**
```bash
# Option A: Set PYTHONPATH
export PYTHONPATH=/path/to/abstractAgentMachine/src

# Option B: Install package
uv sync  # or: pip install -e .
```

**2. ImportError: cannot import name 'HookedTransformer'**

**Solution:**
```bash
uv sync --extra cognitive --extra interpretability
# or: pip install transformer-lens
```

**3. CUDA out of memory (Phase 3)**

**Solutions:**
- Use a smaller model (e.g., `gpt2` instead of `llama-2-7b`)
- Reduce batch size (if applicable)
- Use CPU: `export CUDA_VISIBLE_DEVICES=""`

**4. HuggingFace model download fails**

**Solutions:**
- Check internet connection
- Verify model ID is correct
- For gated models: `huggingface-cli login`
- Check disk space: `df -h ~/.cache/huggingface`

**5. llama.cpp server won't start**

**Solutions:**
- Check if port 8081 is available: `lsof -i :8081`
- Verify model file exists: `ls -lh models/your-model.gguf`
- Check build: `cd third_party/llama.cpp && make llama-server`

**6. Database locked errors**

**Solutions:**
- Ensure only one process writes to the database
- Check for stale lock files: `rm *.db-wal *.db-shm`
- Use WAL mode (default): `PRAGMA journal_mode=WAL;`

**7. Activation files not created (Phase 3)**

**Solutions:**
- Verify `trigger_actions` includes actions agents actually take
- Check that model supports TransformerLens (not all models do)
- Ensure `--layers` and `--components` are valid (use `--list-hooks`)
- Check disk space in `runs/` directory

### Getting Help

1. **Check logs:** Look for error messages in terminal output
2. **Verify installation:** Run health check above
3. **Check dependencies:** `pip list | grep -E "(langgraph|litellm|transformer|torch)"`
4. **Open an issue:** Include error messages and environment details

### Debug Mode

For verbose output, you can add logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export PYTHONUNBUFFERED=1
export AAM_DEBUG=1
```

## Next Steps

After setup, see:
- [README.md](README.md) for usage examples
- [PHASE1_ACCOMPLISHMENTS.md](PHASE1_ACCOMPLISHMENTS.md) for implementation details
- [Abstract Agent Machine PRD.txt](Abstract Agent Machine PRD.txt) for full requirements (Vivarium project)

