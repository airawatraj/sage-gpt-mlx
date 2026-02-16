import sys
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from pathlib import Path

# --- Configuration (STRICT ALIGNMENT WITH YOUR train_engine_mlx.py) ---
VOCAB_SIZE = 12000
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
CONTEXT_LENGTH = 256

CHECKPOINT_PATH = "3-model/mlx/checkpoints/epoch_15.safetensors"
TOKENIZER_PATH = "2-tokenizer/sutra_tokenizer.model" 

# --- Model Architecture (EXACT REPLICA) ---
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

# --- Inference Logic ---
def generate(model, tokenizer, prompt, max_tokens=100, temp=0.8):
    ids = tokenizer.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    print(f"\n>> {prompt}", end="", flush=True)

    for _ in range(max_tokens):
        logits = model(x[:, -CONTEXT_LENGTH:])[:, -1, :]
        if temp > 0:
            token = mx.random.categorical(logits / temp, num_samples=1)
        else:
            token = mx.argmax(logits, axis=-1, keepdims=True)
        
        print(tokenizer.decode([token.item()]), end="", flush=True)
        x = mx.concatenate([x, token], axis=1)
    print("\n")

def main():
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    
    if not Path(CHECKPOINT_PATH).exists():
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}"); return

    try:
        model.load_weights(CHECKPOINT_PATH)
        mx.eval(model.parameters())
        print("✅ Epoch 15 Brain Engaged (512-dim | RoPE).")
    except Exception as e:
        print(f"❌ Load Failed: {e}"); return

    while True:
        p = input("Prompt (Sanskrit) > ")
        if p.lower() in ['q', 'exit']: break
        if p.strip(): generate(model, sp, p)

if __name__ == "__main__":
    main()