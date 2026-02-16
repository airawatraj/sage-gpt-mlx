import os
import sys
import json
import re
import hashlib
import unicodedata
import fitz  # pymupdf
from tqdm import tqdm
from pathlib import Path

# Add project root to path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: duplicate config.py not found. Please run setup_factory.py first.")
    sys.exit(1)

# Constants
PROCESSED_FILES_JSON = config.META_DATA_DIR / "processed_files.json"
CORPUS_FILE = config.PURIFIED_DATA_DIR / "corpus.txt"

# Regex Patterns
PUA_PATTERN = re.compile(r'[\uE000-\uF8FF]')
DIGITS_PATTERN = re.compile(r'[0-9\u0966-\u096F]')
BRACKETS_PATTERN = re.compile(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}')
ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')
NUKTAS_PATTERN = re.compile(r'[\u0958\u0933]') # क़ and ळ
VALID_CHARS_PATTERN = re.compile(r'[^\u0900-\u097F\s\u200c\u200d\u0964\u0965]') # Basic Devanagari range + space + ZWJ/ZWNJ + Danda + Double Danda check inverse
# Note on VALID_CHARS: The user demanded "Pure Devanagari (\u0900-\u097F)".
# Danda is \u0964, Double Danda is \u0965. These are crucial for Sanskrit.
# User said "Pure Devanagari (\u0900-\u097F)". This range INCLUDES dandas (0964, 0965).
# So strict check: if any char is NOT in 0900-097F or space, reject.
# Explicitly allowing Dandas in the comment and logic for verification stability.
STRICT_DEVANAGARI_PATTERN = re.compile(r'[^\u0900-\u097F\s\u0964\u0965]') 

SIGNATURE_PATTERN = re.compile(r'[\u094D\u0903]') # Virama (Halant) or Visarga

def load_state():
    processed_files = set()
    existing_hashes = set()

    if PROCESSED_FILES_JSON.exists():
        with open(PROCESSED_FILES_JSON, "r") as f:
            try:
                processed_files = set(json.load(f))
            except json.JSONDecodeError:
                pass
    
    # Load hashes from existing corpus to avoid duplicates if appending
    # Or just start fresh if we want to ensure total uniqueness for this run.
    # To support "resume", we should ideally load hashes. 
    # Reading 13GB corpus linearly to hash is slow.
    # For now, let's assume we append and just track processed files.
    # But user asked for "Deduplication: Use SHA-256 to ensure every shloka is unique".
    # If we don't load existing hashes, we might duplicate if we re-process a file or if the same verse is in a NEW file.
    # Let's simple-check: IF corpus exists, we read it once to build the hash set.
    if CORPUS_FILE.exists():
        print("Loading existing corpus hashes for deduplication...")
        with open(CORPUS_FILE, "r") as f:
            for line in tqdm(f, desc="Hashing existing corpus"):
                if line.strip():
                    hashed = hashlib.sha256(line.strip().encode()).hexdigest()
                    existing_hashes.add(hashed)

    return processed_files, existing_hashes

def save_state(processed_files):
    with open(PROCESSED_FILES_JSON, "w") as f:
        json.dump(list(processed_files), f, indent=2)

HINDI_STOPWORDS = {"है", "को", "का", "में", "से", "हुए", "थी", "कयने", "कयती", "तथा"}

def is_valid_line(line):
    # 0. Length Check (Density Guard)
    if len(line.replace(" ", "")) < 10:
        return False
        
    # 1. Hindi 'Banish' List
    words = set(line.split())
    if not words.isdisjoint(HINDI_STOPWORDS):
        return False

    # 2. NFKC Normalized (Already done) / Gatekeeper Checks
    
    # Must have NO English
    if ENGLISH_PATTERN.search(line):
        return False

    # Must have NO Nuktas or Modern Noise
    if NUKTAS_PATTERN.search(line):
        return False
        
    # Must have at least one Sanskrit Signature
    if not SIGNATURE_PATTERN.search(line):
        return False

    # Strict Devanagari Check
    if STRICT_DEVANAGARI_PATTERN.search(line):
        return False
        
    # 3. Density Guard (>70% Devanagari)
    devanagari_count = sum(1 for c in line if '\u0900' <= c <= '\u097F')
    total_chars = len(line) # Using total length including spaces/dandas for conservative density
    if total_chars == 0: return False
    
    if (devanagari_count / total_chars) < 0.7:
        return False
        
    return True

def clean_line(line):
    # NFKC Normalization
    line = unicodedata.normalize('NFKC', line)
    
    # Remove PUA
    line = PUA_PATTERN.sub('', line)
    
    # Remove Digits
    line = DIGITS_PATTERN.sub('', line)
    
    # Remove Brackets
    line = BRACKETS_PATTERN.sub('', line)
    
    line = line.strip()
    
    # --- De-Echo Logic & Punctuation Collapse ---
    
    # 1. Punctuation Collapse: consecutive Dandas or Double Dandas
    # Collapses '||||' -> '||', '||' -> '||', '| |' -> '|'
    # Actually, standardizing:
    #   Collapse multiple | or || into single occurences? 
    #   User said: "consecutive Dandis (॥॥) or terminal 'm's (म्म्) are collapsed into single characters"
    #   So ॥॥ -> ॥, || -> |, म्म् -> म्.
    
    line = re.sub(r'([।॥])\1+', r'\1', line)
    line = re.sub(r'(म्)\1+', r'\1', line)
    
    # 2. De-Echo Suffixes (Repeated patterns at end of string)
    # Target: 'यैायैनमः' -> 'यै नमः' (This is specific 'Ya-Ai-aa-Ya-Ai-Namah' -> 'Ya-Ai Namah')
    # Use a regex for general repeated groups of 2+ chars at the end
    # Matches (group of 2+ chars) followed immediately by itself at the end of the line
    line = re.sub(r'(\S{2,})\1$', r'\1', line)
    
    # Specific fix for the user's example if general regex fails
    # 'यैायैनमः' isn't a direct doubling of 'यै', it has 'ा' in between or is 'यै' 'ा' 'यै'.
    # If the user meant "yayai" -> "yai", that's handled.
    # If they strictly meant "yai-aa-yai" -> "yai", that's `(\S+)\u093e\1` -> `\1`.
    # Let's add a generic adjacent word deduper for the end of line:
    # e.g. "word word" -> "word" at EOL.
    line = re.sub(r'(\S+)\s+\1$', r'\1', line)
    
    return line

def process_file(pdf_path, existing_hashes, output_file_handle):
    unique_count = 0
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text("text")
            for line in text.split('\n'):
                # 1. Surgical Cleaning
                cleaned = clean_line(line)
                
                # 2. Linguistic Gatekeeper
                if not cleaned:
                    continue
                    
                if is_valid_line(cleaned):
                    # 3. Deduplication
                    line_hash = hashlib.sha256(cleaned.encode()).hexdigest()
                    if line_hash not in existing_hashes:
                        output_file_handle.write(cleaned + '\n')
                        existing_hashes.add(line_hash)
                        unique_count += 1
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
    
    return unique_count

def main():
    print("Starting Visuddhi (Data Cleaning)...")
    
    # Ensure raw directory has files (Deep Recursion)
    print(f"Scanning for PDFs in {config.RAW_DATA_DIR}...")
    raw_files = sorted(list(config.RAW_DATA_DIR.rglob("*.pdf")))
    
    print(f"Total PDFs found: {len(raw_files)}")
    
    if not raw_files:
        print(f"No PDF files found in {config.RAW_DATA_DIR}")
        return

    processed_files, existing_hashes = load_state()
    
    files_to_process = [f for f in raw_files if f.name not in processed_files]
    
    if not files_to_process:
        print("All files already processed.")
        return

    print(f"Found {len(files_to_process)} new files to process.")
    
    # Open corpus file in append mode
    with open(CORPUS_FILE, "a") as out_f:
        # Progress Bar: [PDFs | Unique Verses | Current Rate]
        pbar = tqdm(files_to_process, unit="pdf")
        total_unique = len(existing_hashes)
        
        for file_path in pbar:
            new_unique = process_file(file_path, existing_hashes, out_f)
            total_unique += new_unique
            processed_files.add(file_path.name)
            
            # Update desc
            pbar.set_description(f"PDFs: {len(processed_files)}/{len(raw_files)} | Unique: {total_unique}")
            
            # Save state incrementally (or per file to be safe)
            save_state(processed_files)

    print("Visuddhi Complete.")
    print(f"Total Unique Verses: {len(existing_hashes)}")

if __name__ == "__main__":
    main()
