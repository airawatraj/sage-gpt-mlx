import sys
import sentencepiece as spm
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

MODEL_PATH = config.TOKENIZER_DIR / "sutra_tokenizer.model"

def main():
    print(f"Loading Sutra Tokenizer from {MODEL_PATH}...")
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run sutra_tokenizer.py first.")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))
    
    print("-" * 50)
    print("SUTRA PROBE: Interactive Tokenizer Diagnostic")
    print("Type text to visualize tokens. Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nInput > ").strip()
            if text.lower() in ('exit', 'quit'):
                break
            if not text:
                continue
                
            # 1. IDs
            ids = sp.encode_as_ids(text)
            print(f"\n[IDs]: {ids}")
            
            # 2. Pieces
            pieces = sp.encode_as_pieces(text)
            print(f"[Pieces]: {pieces}")
            
            # 3. Reconstruction
            decoded = sp.decode(ids)
            print(f"[Reconstruction]: {decoded}")
            
            # Verification
            if decoded == text:
                print("[Status]: ✅ Lossless")
            else:
                print(f"[Status]: ❌ Lossy (Diff: {decoded!r})")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nExiting Sutra Probe.")

if __name__ == "__main__":
    main()
