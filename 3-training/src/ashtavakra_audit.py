
import sys
import os
import time
import math
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from pathlib import Path

# --- Configuration & Paths ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
    CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
    TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
except ImportError:
    # Fallback if config not found
    CHECKPOINT_DIR = project_root / "3-model/mlx/checkpoints"
    TOKENIZER_MODEL = project_root / "2-tokenizer/sutra_tokenizer.model"

INTERRUPT_SAVE = CHECKPOINT_DIR / "interrupt_save.safetensors"

# Architecture (Must match train_engine_mlx.py)
VOCAB_SIZE = 8000
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
CONTEXT_LENGTH = 512
DROPOUT = 0.1

# --- Model Classes (EXACT REPLICA of train_engine_mlx.py) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_embd // n_head) ** -0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(n_embd // n_head, traditional=True)
        self.dropout = nn.Dropout(DROPOUT) # Added Dropout

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        q = self.rope(q); k = self.rope(k)
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None: att = att + mask
        att = mx.softmax(att, axis=-1)
        att = self.dropout(att) # Added Dropout application
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT) # Added Dropout
        )
    def __call__(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = FeedForward(n_embd)
        self.dropout = nn.Dropout(DROPOUT) # Added Dropout
    def __call__(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask)) # Added Dropout application
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
        x = self.embedding(x)
        for block in self.blocks: x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# --- Helpers ---

def load_hot_sync(model):
    """Loads weights from interrupt_save or latest epoch, readonly."""
    ckpt_path = None
    if INTERRUPT_SAVE.exists():
        ckpt_path = INTERRUPT_SAVE
    else:
        # Sort by epoch number
        checkpoints = sorted(CHECKPOINT_DIR.glob("epoch_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]), reverse=True)
        if checkpoints:
            ckpt_path = checkpoints[0]
    
    if ckpt_path:
        print(f"🔥 Hot-Sync: Loading wisdom from {ckpt_path.name}...")
        try:
            # Use mx.load to load weights (read-only safe)
            model.load_weights(str(ckpt_path))
            return True
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return False
    else:
        print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
        return False

def generate(model, tokenizer, prompt, max_tokens=10, temp=0.0):
    ids = tokenizer.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    generated_ids = []
    
    for _ in range(max_tokens):
        logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            token = mx.random.categorical(logits * (1/temp), num_samples=1)
        
        token_item = token.item()
        generated_ids.append(token_item)
        x = mx.concatenate([x, token.reshape(1,1)], axis=1)
        
    return tokenizer.decode(generated_ids)

def get_next_token_probs(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
    probs = mx.softmax(logits[0])
    return probs

def is_sanskrit(text):
    """Checks if text contains Devanagari characters."""
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return True
    return False

# --- The 8 Bends (Hybrid Edition) ---

# STATUS Constants (English Only for alignment)
STRAIGHT = "STRAIGHT"
CROOKED  = "CROOKED"
WOBBLY   = "WOBBLY"

def check_stutter(text):
    """Checks for repeating 4-char sequences > 2 times."""
    if len(text) < 8: return False
    for i in range(len(text) - 4):
        gram = text[i:i+4]
        if text.count(gram) > 2:
            return True
    return False

def bend_1_phonetics(model, tokenizer):
    """1. Phonetic Stability: ॐ -> Flow (Checking Stutter)."""
    prompt = "ॐ"
    gen = generate(model, tokenizer, prompt, max_tokens=30, temp=0.1)
    
    # Stutter Check
    if check_stutter(gen):
        return WOBBLY, f"Stuttering: '{gen[:50]}...'"
        
    # Relaxed Logic: If output contains Sanskrit chars beyond prompt, mark STRAIGHT
    if is_sanskrit(gen):
         return STRAIGHT, f"Sanskrit Flow: '{gen[:50]}...'"
    
    if len(gen) > 5:
         return CROOKED, f"No Sanskrit: '{gen[:50]}...'"
    
    # Fallback
    return STRAIGHT, f"Valid: '{gen[:50]}...'"

def bend_2_invocation(model, tokenizer):
    """2. Invocation: असतो मा -> सद्गमय."""
    prompt = "असतो मा"
    gen = generate(model, tokenizer, prompt, max_tokens=30, temp=0.1)
    
    if check_stutter(gen):
        return WOBBLY, f"Stuttering: '{gen[:50]}...'"

    # Check for keywords
    if "सद्गमय" in gen or "तमसो" in gen:
        return STRAIGHT, f"Resonance: '{gen[:50]}...'"

    return CROOKED, f"Missed: '{gen[:50]}...'"

def bend_3_vibhakti(model, tokenizer):
    """3. Case Inflection: राम -> P(ः) > 0.4."""
    prompt = "राम"
    # Identify ID for "ः" from 8000 vocab
    target_ids = tokenizer.encode("ः")
    if not target_ids or len(target_ids) != 1:
        return CROOKED, f"ID Error: 'ः' -> {target_ids}"
    target_id = target_ids[0]
    
    probs = get_next_token_probs(model, tokenizer, prompt)
    p_visarga = probs[target_id].item()
    
    if p_visarga > 0.4:
        return STRAIGHT, f"P(ः) = {p_visarga:.2f}"
    return CROOKED, f"P(ः) = {p_visarga:.2f} (Low)"

def bend_4_guna_sandhi(model, tokenizer):
    """4. Name Synthesis: नर -> इन्द्रः -> नरेन्द्रः."""
    prompt = "नर"
    gen = generate(model, tokenizer, prompt, max_tokens=5, temp=0)
    
    if "इन्द्र" in gen or "ेन्द्र" in gen:
        return STRAIGHT, f"Completed: ...{gen[:10]}"
    return CROOKED, f"Got: '{gen[:10]}...'"

def bend_5_concept_flow(model, tokenizer):
    """5. Concept Flow: यथा नद्यः -> स्यन्दमानाः/समुद्रे."""
    prompt = "यथा नद्यः"
    gen = generate(model, tokenizer, prompt, max_tokens=30, temp=0.1)
    
    if check_stutter(gen):
        return WOBBLY, f"Stuttering: '{gen[:50]}...'"
    
    if "समुद्रे" in gen or "स्यन्दमाना" in gen:
        return STRAIGHT, f"Flowing: ...{gen[:50]}..."
    return CROOKED, f"Blocked: '{gen[:50]}...'"

def bend_6_verse_isha(model, tokenizer):
    """6. Verse Sequence: ईशा वास्यमिदं -> सर्वं."""
    prompt = "ईशा वास्यमिदं"
    gen = generate(model, tokenizer, prompt, max_tokens=30, temp=0.1)
    
    if check_stutter(gen):
        return WOBBLY, f"Stuttering: '{gen[:50]}...'"
    
    if "सर्वं" in gen or "जगत्" in gen or "यत्किञ्च" in gen:
        return STRAIGHT, f"Correct verse: ...{gen[:50]}..."
    return CROOKED, f"Lost: '{gen[:50]}...'"

def bend_7_orthography(model, tokenizer):
    """7. Orthography: कृष् -> ण -> कृष्ण."""
    prompt = "कृष्"
    gen = generate(model, tokenizer, prompt, max_tokens=3, temp=0)
    
    if "ण" in gen:
        return STRAIGHT, "Correct cluster (ण)"
    return CROOKED, f"Broken cluster: '{gen[:10]}...'"

def bend_8_atman(model, tokenizer):
    """8. The Atman Test: तत्त्वमसि -> श्वेतकेतो."""
    prompt = "तत्त्वमसि"
    gen = generate(model, tokenizer, prompt, max_tokens=10, temp=0)
    
    if "श्वेतकेतो" in gen:
        return STRAIGHT, "Resonates (Shvetaketo)"
    return CROOKED, f"Dissonant: '{gen[:10]}...'"

# --- Main ---

def main():
    print("\n🐚 Ashtavakra Audit: Hybrid Diagnostic Hub 🐚")
    print("=============================================")
    
    # 1. Setup
    if not TOKENIZER_MODEL.exists():
        print(f"❌ Tokenizer not found at {TOKENIZER_MODEL}")
        return
        
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    
    # 2. Load
    if not load_hot_sync(model): return
    
    # 3. The Scorecard (ASCII ONLY for Perfect Alignment)
    # Removing Sanskrit from Column 1 prevents 'ligature collapse' alignment bugs.
    
    print("-" * 90) 
    print(f"{'BEND (TEST)':<30} | {'STATUS':<10} | {'DETAILS':<45}")
    print("-" * 90)
    
    tests = [
        ("1. Phonetic Stability", bend_1_phonetics),    # Removed (ॐ)
        ("2. Invocation", bend_2_invocation),          # Removed (असतो मा)
        ("3. Case Inflection", bend_3_vibhakti),       # Removed (रामः)
        ("4. Name Synthesis", bend_4_guna_sandhi),     # Removed (नरेन्द्रः)
        ("5. Concept Flow", bend_5_concept_flow),      # Removed (यथा नद्यः)
        ("6. Verse Sequence", bend_6_verse_isha),      # Removed (ईशा)
        ("7. Orthography", bend_7_orthography),        # Removed (कृष्णः)
        ("8. The Atman Test", bend_8_atman),
    ]
    
    score = 0
    for name, func in tests:
        # Pass model and tokenizer to each test function
        try:
            status, details = func(model, sp)
        except Exception as e:
            status = CROOKED
            details = f"Error: {e}"

        if status == STRAIGHT: score += 1
        
        # Details Truncation: Hard limit to prevent wrapping
        clean_details = details.replace('\n', ' ') # Remove newlines if any
        if len(clean_details) > 43:
            clean_details = clean_details[:41] + ".."
            
        print(f"{name:<30} | {status:<10} | {clean_details}")
    
    print("-" * 90)
    print(f"🕸️  Vedic Scorecard: {score}/8 Bends Straightened")
    print("\n")

if __name__ == "__main__":
    main()
