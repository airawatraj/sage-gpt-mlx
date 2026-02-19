import sys
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from pathlib import Path

VOCAB_SIZE, N_LAYER, N_HEAD, N_EMBD, CONTEXT_LENGTH = 8000, 6, 6, 384, 512

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_embd // n_head) ** -0.5
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(n_embd // n_head, traditional=True)

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None: att = att + mask
        y = (mx.softmax(att, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.c_proj(y)

class TransformerLM(nn.Module):
    def __init__(self, v, l, e, h):
        super().__init__()
        self.embedding = nn.Embedding(v, e)
        self.blocks = [nn.Sequential(nn.RMSNorm(e), MultiHeadAttention(e, h)) for _ in range(l)]
        self.ln_f = nn.RMSNorm(e)
        self.head = nn.Linear(e, v, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        x = self.embedding(x)
        for b in self.blocks: x = x + b(x, mask)
        return self.head(self.ln_f(x))

def sample_top_p(logits, temp=0.8, top_p=0.9, rep_pen=1.2, seen_ids=None):
    if seen_ids:
        indices = mx.array(list(set(seen_ids)))
        logits[indices] -= rep_pen
    logits /= max(temp, 1e-5)
    sorted_idx = mx.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]
    mask = mx.cumsum(mx.softmax(sorted_logits)) > top_p
    mask[0] = False
    sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)
    return sorted_idx[mx.random.categorical(mx.softmax(sorted_logits))]

def main():
    sp = spm.SentencePieceProcessor(model_file="2-tokenizer/sutra_tokenizer.model")
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    model.load_weights("3-model/mlx/checkpoints/interrupt_save.safetensors")
    mx.eval(model.parameters())
    
    while True:
        p = input("\nPrompt > ")
        if p.lower() in ["q", "exit"]: break
        for mode in ["RAW", "SAMPLED"]:
            print(f"\n[{mode}]: ", end="", flush=True)
            ids = sp.encode(p)
            for _ in range(50 if mode=="RAW" else 100):
                logits = model(mx.array([ids]))[0, -1]
                tid = mx.argmax(logits).item() if mode=="RAW" else sample_top_p(logits, seen_ids=ids).item()
                print(sp.decode([tid]), end="", flush=True); ids.append(tid)
            print()

if __name__ == "__main__": main()
