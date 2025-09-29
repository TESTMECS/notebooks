# 📓 Notebooks

A collection of research notebooks exploring machine learning ideas.

## 📁 Project Structure

`pub/` -> notebooks that I feel like stand on their own.

## 🚀 Quick Start

### Minimal Installation (CLI + Server only)
```bash
# Install just the essentials for CLI and server
pip install -e .
# Or with uv (recommended)
uv sync 
```

This installs only:
- `rich` - For beautiful CLI output
- `watchdog` - For file watching

### Optional Dependencies

#### For Machine Learning Notebooks
```bash
# Install ML dependencies (torch, scikit-learn, etc.)
uv sync --extra ml 

or 

pip install -e ".[ml]"
```

#### For Marimo Notebooks
```bash
# Install marimo for interactive notebooks
uv sync --extra notebooks

or

pip install -e ".[notebooks]"
```

#### For GUI Components
```bash
# Install PyQt5 for GUI notebooks
uv sync --extra gui

or 

pip install -e ".[gui]"
```

#### For AI/LLM Features
```bash
# Install Google Generative AI
uv sync --extra ai

or

pip install -e ".[ai]"
```

#### Everything
```bash
# Install all optional dependencies
pip install -e ".[full]"
```
