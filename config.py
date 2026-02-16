# Auto-generated config by setup_factory.py
from pathlib import Path

PROJECT_ROOT = Path("/Users/rajrawat/rawatlabs/sage-gpt")

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
