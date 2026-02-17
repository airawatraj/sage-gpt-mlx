
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
except ImportError:
    PROJECT_ROOT = Path(".").resolve()

# Architecture (Must match train_engine_mlx.py)
VOCAB_SIZE = 12000
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
CONTEXT_LENGTH = 256

# Paths
CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"
INTERRUPT_SAVE = CHECKPOINT_DIR / "interrupt_save.safetensors"

# --- Model Classes ---
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_embd // n_head) ** -0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(n_embd // n_head)

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
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def __call__(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)
    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
        x = self.embedding(x)
        for block in self.blocks: x = block(x, mask)
        return self.head(self.ln_f(x))

# --- Helpers ---

def load_hot_sync(model):
    """Loads weights from interrupt_save or latest epoch, readonly."""
    ckpt_path = None
    if INTERRUPT_SAVE.exists():
        ckpt_path = INTERRUPT_SAVE
    else:
        checkpoints = list(CHECKPOINT_DIR.glob("epoch_*.safetensors"))
        if checkpoints:
            ckpt_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    
    if ckpt_path:
        print(f"🔥 Hot-Sync: Loading wisdom from {ckpt_path.name}...")
        try:
            model.load_weights(str(ckpt_path))
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
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

# --- The 8 Bends (Hybrid Edition) ---

STRAIGHT = "STRAIGHT (ऋजु)"
CROOKED = "CROOKED (वक्र)"

def bend_1_phonetics(model, tokenizer):
    """1. Phonetic Stability: ॐ -> Varied."""
    prompt = "ॐ"
    gen = generate(model, tokenizer, prompt, max_tokens=10, temp=0.1)
    
    loops = ["ानां", "ााा", "ननन", "ततत"]
    for l in loops:
        if l in gen:
            return CROOKED, f"Loop detected: '{l}'"
            
    distinct_chars = set(gen)
    if len(gen) > 5 and len(distinct_chars) < 3:
         return CROOKED, f"Low variety: '{gen}'"
    
    # Pass if varied
    return STRAIGHT, f"Varied output: '{gen}'"

def bend_2_boundary(model, tokenizer):
    """2. Boundary/Spacing: ॐ -> Space or Newline."""
    prompt = "ॐ"
    probs = get_next_token_probs(model, tokenizer, prompt)
    top_id = mx.argmax(probs).item()
    top_tok = tokenizer.decode([top_id])
    
    if top_tok.startswith(" ") or top_tok.startswith("\n") or top_tok == "":
        return STRAIGHT, f"Boundary found: '{top_tok}'"
    return CROOKED, f"Merged/No space: '{top_tok}'"

def bend_3_vibhakti(model, tokenizer):
    """3. Case Inflection: राम -> P(ः) > 0.4."""
    prompt = "राम"
    # Identify ID for "ः"
    target_ids = tokenizer.encode("ः")
    if not target_ids: return CROOKED, "Tokenizer Error"
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
        return STRAIGHT, f"Completed: ...{gen}"
    return CROOKED, f"Got: '{gen}'"

def bend_5_dirgha_sandhi(model, tokenizer):
    """5. Concept Merger: हिम -> आलयः -> हिमालयः."""
    prompt = "हिम"
    gen = generate(model, tokenizer, prompt, max_tokens=5, temp=0)
    
    if "आलय" in gen or "ालय" in gen:
        return STRAIGHT, f"Completed: ...{gen}"
    return CROOKED, f"Got: '{gen}'"

def bend_6_verse(model, tokenizer):
    """6. Verse Sequence: धर्मक्षेत्रे कुरुक्षेत्रे -> समवेता."""
    prompt = "धर्मक्षेत्रे कुरुक्षेत्रे"
    gen = generate(model, tokenizer, prompt, max_tokens=5, temp=0)
    
    if "समवेता" in gen:
        return STRAIGHT, "Correct continuation"
    return CROOKED, f"Got: '{gen}'"

def bend_7_orthography(model, tokenizer):
    """7. Orthography: कृष् -> ण -> कृष्ण."""
    prompt = "कृष्"
    gen = generate(model, tokenizer, prompt, max_tokens=3, temp=0)
    
    if "ण" in gen:
        return STRAIGHT, "Correct cluster (ण)"
    return CROOKED, f"Broken cluster: '{gen}'"

def bend_8_atman(model, tokenizer):
    """8. The Atman Test: तत्त्वमसि -> श्वेतकेतो."""
    prompt = "तत्त्वमसि"
    gen = generate(model, tokenizer, prompt, max_tokens=10, temp=0)
    
    if "श्वेतकेतो" in gen:
        return STRAIGHT, "Resonates (Shvetaketo)"
    return CROOKED, f"Dissonant: '{gen}'"

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
    
    print("-" * 75)
    print(f"{'BEND (TEST)':<30} | {'STATUS':<15} | {'DETAILS'}")
    print("-" * 75)
    
    tests = [
        ("1. Phonetic Stability (Om)", bend_1_phonetics),
        ("2. Boundary/Spacing", bend_2_boundary),
        ("3. Case Inflection (Ramah)", bend_3_vibhakti),
        ("4. Name Synthesis (Narendra)", bend_4_guna_sandhi),
        ("5. Concept Merger (Himalaya)", bend_5_dirgha_sandhi),
        ("6. Verse Sequence (Gita)", bend_6_verse),
        ("7. Orthography (Krishna)", bend_7_orthography),
        ("8. The Atman Test", bend_8_atman),
    ]
    
    score = 0
    for name, func in tests:
        status, details = func(model, sp)
        if status == STRAIGHT: score += 1
        print(f"{name:<30} | {status:<15} | {details}")
    
    print("-" * 75)
    print(f"🕸️  Vedic Scorecard: {score}/8 Bends Straightened")
    print("\n")

if __name__ == "__main__":
    main()
