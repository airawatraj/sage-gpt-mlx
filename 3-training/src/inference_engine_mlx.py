import sys
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from pathlib import Path

# --- Configuration (STRICT ALIGNMENT WITH train_engine_mlx.py) ---
VOCAB_SIZE = 8000
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
CONTEXT_LENGTH = 512
DROPOUT = 0.1

# --- Paths Setup ---
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

# --- Model Architecture (EXACT REPLICA of train_engine_mlx.py) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_embd // n_head) ** -0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(n_embd // n_head, traditional=True)
        self.dropout = nn.Dropout(DROPOUT)

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        
        q = self.rope(q)
        k = self.rope(k)
        
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            att = att + mask
        att = mx.softmax(att, axis=-1)
        att = self.dropout(att)
        
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )
    def __call__(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = FeedForward(n_embd)
        self.dropout = nn.Dropout(DROPOUT)
    def __call__(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
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
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# --- Inference Logic ---
def generate(model, tokenizer, prompt, max_tokens=100, temp=0.8):
    ids = tokenizer.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    print(f"\n>> {prompt}", end="", flush=True)

    for _ in range(max_tokens):
        logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
        
        if temp > 0:
            token = mx.random.categorical(logits * (1.0 / temp), num_samples=1)
        else:
            token = mx.argmax(logits, axis=-1, keepdims=True)
        
        # item() is robust for single scalar extraction from MLX array
        token_id = token.item()
        
        print(tokenizer.decode([token_id]), end="", flush=True)
        x = mx.concatenate([x, token], axis=1)
    print("\n")

def main():
    print("SanskritGPT Inference Engine (MLX)")
    
    if not TOKENIZER_MODEL.exists():
        print(f"❌ Tokenizer not found at {TOKENIZER_MODEL}")
        return

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    
    # Try different potential checkpoint locations
    possible_ckpts = [
        CHECKPOINT_DIR / "interrupt_save.safetensors",
    ]
    # Add numbered epochs
    possible_ckpts.extend(sorted(CHECKPOINT_DIR.glob("epoch_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]), reverse=True))
    
    loaded_path = None
    for p in possible_ckpts:
        if p.exists():
            loaded_path = p
            break
            
    if not loaded_path:
        print(f"❌ No checkpoints found in {CHECKPOINT_DIR}")
        return

    try:
        model.load_weights(str(loaded_path))
        mx.eval(model.parameters())
        print(f"✅ Loaded weights from {loaded_path.name}")
        print(f"✅ Architecture Match: VOCAB={VOCAB_SIZE}, CTX={CONTEXT_LENGTH}, Dropout={DROPOUT}")
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    print("\nReady. Type 'exit' or 'q' to quit.")
    while True:
        try:
            p = input("Prompt (Sanskrit) > ")
            if p.lower() in ['q', 'exit']: break
            if p.strip(): generate(model, sp, p)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
