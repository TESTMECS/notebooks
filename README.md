# 📓 Notebooks

A collection of research notebooks exploring machine learning ideas.

## 📁 Project Structure

`pub/` -> notebooks that I feel like stand on their own. You can check out the HTML versions here: [link](https://TESTMECS.github.io/)

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

## 🖥️ Usage

### Start the Server
```bash
# From the project root
python -m backend.main
# or
cd backend && python main.py
```

### Use the CLI
```bash
# Interactive mode
python main.py

# List notebooks
python main.py --list

# Open Public webpage.
python main.py --view '<path/to/nb.py>'

# Show images
python main.py --images

# Get inspiring quotes
python main.py --quote
```

## Run Marimo Notebooks
```bash
# Marimo notebooks are self-contained with their dependencies
marimo run pub/fractalInterpFunctions.py
```
### Useful flags. 
`--sandbox` -> Only works with UV. Dependencies are declared in comment at top of file and installed upon loading the notebook.  

## 💡 Philosophy

Each marimo notebook includes its own dependency specification at the top (script tag), allowing for isolated environments per notebook. This means:

- ✅ No need to install heavy ML libraries just to run the server
- ✅ Marimo handles sandboxed environments automatically  
- ✅ Fast startup times for basic operations
- ✅ Each notebook specifies exactly what it needs

---
Created with <3 with [Marimo](https://marimo.io/)!
