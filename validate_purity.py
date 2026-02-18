import re
import sys
from collections import Counter

# Strict Definition: Devanagari Unicode Block + Common Punctuation
# \u0900-\u097F: Devanagari
# \s: Whitespace
# \u200c\u200d: ZWJ/ZWNJ (Allowed in Sanskrit typesetting)
VALID_CHARS = re.compile(r'[^\u0900-\u097F\s\u200c\u200d\u0964\u0965]')

def validate(filepath):
    print(f"Validating Purity: {filepath}")
    bad_chars = Counter()
    total_chars = 0
    line_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            total_chars += len(line)
            
            # Find invalid chars
            invalids = VALID_CHARS.findall(line)
            if invalids:
                bad_chars.update(invalids)
                
    print(f"Scanned {line_count} lines ({total_chars} chars).")
    
    if not bad_chars:
        print("✅ SUCCESS: 100% Pure Sanskrit (Devanagari Only).")
    else:
        print(f"❌ WARNING: Found {sum(bad_chars.values())} invalid characters.")
        print(f"Top Offenders: {bad_chars.most_common(10)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate(sys.argv[1])
