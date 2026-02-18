import os
import sys
import re
import fitz  # pymupdf
import unicodedata
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import config

# Configuration
INPUT_DIR = config.RAW_DATA_DIR
OUTPUT_FILE = config.PURIFIED_DATA_DIR / "corpus.txt"
MIN_BLOCK_LENGTH = 15  # Minimum continuous Devanagari characters to keep
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F\s\u200c\u200d\u0964\u0965]+')

def clean_text_block(text):
    """
    Extracts only valid Devanagari blocks from text.
    """
    # Normalize
    text = unicodedata.normalize('NFKC', text)
    
    # Find all Devanagari sequences
    matches = DEVANAGARI_PATTERN.findall(text)
    
    valid_blocks = []
    for m in matches:
        # Filter by length and density
        cleaned = re.sub(r'\s+', ' ', m).strip()
        if len(cleaned) >= MIN_BLOCK_LENGTH:
            valid_blocks.append(cleaned)
            
    return "\n".join(valid_blocks)

def process_html_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            return clean_text_block(text)
    except Exception as e:
        return ""

def process_pdf_file(filepath):
    try:
        doc = fitz.open(filepath)
        full_text = []
        
        for page in doc:
            text = page.get_text()
            cleaned = clean_text_block(text)
            if cleaned:
                full_text.append(cleaned)
        
        return "\n".join(full_text)
    except Exception as e:
        return ""

def main():
    print(f"Starting Data Rescue (Smart Extraction)...")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    
    all_files = []
    # Collect files
    extensions = {'.html', '.htm', '.pdf', '.txt'}
    for ext in extensions:
        all_files.extend(list(INPUT_DIR.rglob(f"*{ext}")))
        
    print(f"Found {len(all_files)} files.")
    
    # Prioritize HTML/TXT as they are likely text-rich
    # But mixed PDFs are the main issue.
    
    total_chars = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        pbar = tqdm(all_files)
        for fp in pbar:
            content = ""
            if fp.suffix.lower() in ['.html', '.htm']:
                content = process_html_file(fp)
                # If HTML fails or is empty, try as plain text?
                # Sometimes html files are just fragments.
            elif fp.suffix.lower() == '.pdf':
                content = process_pdf_file(fp)
            elif fp.suffix.lower() == '.txt':
                try:
                    content = clean_text_block(fp.read_text(encoding='utf-8', errors='ignore'))
                except: pass
            
            if content:
                out_f.write(content + "\n")
                total_chars += len(content)
                pbar.set_description(f"Extracted: {total_chars/1e6:.1f}M chars")

    print(f"Rescue Complete. Total Extracted: {total_chars/1e6:.2f} Million Characters.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
