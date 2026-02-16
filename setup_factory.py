#!/usr/bin/env python3
import os
import sys
import subprocess
import json
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.absolute()
DIRECTORIES = [
    "1-data/01-raw",
    "1-data/02-purified",
    "1-data/03-tokenized",
    "1-data/04-meta",
    "2-tokenizer",
    "3-model",
    "3-training/src",
    "4-output",
]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
CONFIG_FILE = PROJECT_ROOT / "config.py"
STATUS_FILE = PROJECT_ROOT / ".sage_status"
VENV_DIR = PROJECT_ROOT / "sage-gpt"

def log(message):
    print(f"[Factory Setup] {message}")

def create_directories():
    log("Checking directory structure...")
    for dir_path in DIRECTORIES:
        path = PROJECT_ROOT / dir_path
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            log(f"Created: {path}")
        else:
            log(f"Exists: {path}")

def setup_environment():
    log("Checking environment...")
    
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("Error: 'uv' is not installed or not in PATH. Please install uv first.")
        sys.exit(1)

    # Create virtual environment using uv
    if not VENV_DIR.exists():
        log(f"Creating virtual environment '{VENV_DIR.name}' with uv...")
        subprocess.run(["uv", "venv", VENV_DIR.name], check=True, cwd=PROJECT_ROOT)
    else:
        log("Virtual environment already exists.")

    # Install requirements
    if REQUIREMENTS_FILE.exists():
        log("Installing requirements...")
        subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], check=True, cwd=PROJECT_ROOT)
    else:
        log(f"Warning: {REQUIREMENTS_FILE} not found. Skipping dependency installation.")

def generate_config():
    log("Generating config.py...")
    
    config_content = f"""# Auto-generated config by setup_factory.py
from pathlib import Path

PROJECT_ROOT = Path("{PROJECT_ROOT}")

RAW_DATA_DIR = PROJECT_ROOT / "1-data/01-raw"
PURIFIED_DATA_DIR = PROJECT_ROOT / "1-data/02-purified"
TOKENIZED_DATA_DIR = PROJECT_ROOT / "1-data/03-tokenized"
META_DATA_DIR = PROJECT_ROOT / "1-data/04-meta"
TOKENIZER_DIR = PROJECT_ROOT / "2-tokenizer"
MODEL_DIR = PROJECT_ROOT / "3-model"
TRAINING_SRC_DIR = PROJECT_ROOT / "3-training/src"
OUTPUT_DIR = PROJECT_ROOT / "4-output"
VENV_DIR = PROJECT_ROOT / "sage-gpt"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PURIFIED_DATA_DIR, TOKENIZED_DATA_DIR, META_DATA_DIR, TOKENIZER_DIR, MODEL_DIR, TRAINING_SRC_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
"""
    with open(CONFIG_FILE, "w") as f:
        f.write(config_content)
    log(f"Config generated at: {CONFIG_FILE}")

def update_status(stage):
    status = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            try:
                status = json.load(f)
            except json.JSONDecodeError:
                pass
    
    status[stage] = True
    
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)
    log(f"Updated status: {stage} completed.")

def main():
    log("Starting Factory Setup...")
    
    create_directories()
    update_status("directory_structure")
    
    setup_environment()
    update_status("environment_setup")
    
    generate_config()
    update_status("config_generation")
    
    log("Factory Setup Complete!")
    log(f"To activate the environment: source {VENV_DIR}/bin/activate")

if __name__ == "__main__":
    main()
