import sys
import os
import time
import math
import csv
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import sentencepiece as spm
from pathlib import Path
from datetime import datetime, timezone, timedelta

# --- Paths Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    print("Error: config.py not found.")
    sys.exit(1)

# --- Configuration ---
# Architecture
VOCAB_SIZE = 12000
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
CONTEXT_LENGTH = 256

# Training
WEIGHT_DECAY = 0.1
LEARNING_RATE_MAX = 3e-4
WARMUP_STEPS = 1000

# Governor (AEDT UTC+11)
AEDT_OFFSET = timezone(timedelta(hours=11))
STEALTH_BATCH_SIZE = 4
FACTORY_BATCH_SIZE = 32
STEALTH_SLEEP = 0.05
MEMORY_LIMIT_BYTES = 12 * 1024 * 1024 * 1024  # 12 GB
LOWER_MEMORY_LIMIT = 10 * 1024 * 1024 * 1024 # 10 GB target for Stealth

# Paths
DATA_PATH = config.TOKENIZED_DATA_DIR / "corpus.bin"
CHECKPOINT_DIR = config.MODEL_DIR / "mlx" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.OUTPUT_DIR / "logs" / "training_history.csv"
(config.OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = config.TOKENIZER_DIR / "sutra_tokenizer.model"

# --- Logging Setup ---
file_exists = list(LOG_FILE.exists() for _ in [0]) # Hacky check for header
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        writer.writerow(["Timestamp", "Step", "Epoch", "Loss", "Val_Loss", "Mode", "Memory_GB", "Batch_Size", "Tokens_Per_Sec"])

def log_metrics(step, epoch, loss, val_loss, mode, mem_gb, batch_size, tps):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(AEDT_OFFSET).isoformat(), step, epoch, f"{loss:.4f}", val_loss, mode, f"{mem_gb:.2f}", batch_size, f"{tps:.2f}"])

# --- Model Architecture ---

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
        
        q = self.rope(q)
        k = self.rope(k)
        
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            att = att + mask
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
    def __call__(self, x):
        return self.net(x)

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
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# --- Helper Logic ---

def get_governor_state(current_batch_size):
    """
    Returns (target_batch_size, sleep_time, mode_name) based on AEDT time and Memory.
    """
    # 1. Memory Safety Override
    active_mem = mx.metal.get_active_memory()
    if active_mem > MEMORY_LIMIT_BYTES:
        print(f"[SAFETY] Memory {active_mem/1024**3:.2f}GB > Limit. Dropping Batch Size.")
        new_bs = max(1, current_batch_size // 2)
        mx.metal.clear_cache()
        return new_bs, 0.1, "SAFETY_RECOVERY"

    # 2. Time-Based Governor
    now = datetime.now(AEDT_OFFSET)
    hour = now.hour
    is_work_hours = 9 <= hour < 18
    
    if is_work_hours:
        # Stealth Mode
        return STEALTH_BATCH_SIZE, STEALTH_SLEEP, "STEALTH"
    else:
        # Factory Mode
        return FACTORY_BATCH_SIZE, 0.0, "FACTORY"

def generate_cooing(model, tokenizer):
    """Generates 32 tokens from 'ॐ '"""
    input_ids = tokenizer.encode('ॐ ') # [ID]
    x = mx.array([input_ids], dtype=mx.uint32)
    
    tokens = [t for t in input_ids]
    
    for _ in range(32):
        logits = model(x[:, -CONTEXT_LENGTH:])
        logits = logits[:, -1, :]
        token = mx.random.categorical(logits, num_samples=1).item()
        tokens.append(token)
        x = mx.concatenate([x, mx.array([[token]])], axis=1)
    
    decoded = tokenizer.decode(tokens)
    print(f"\n[SAGE-COO]: {decoded}\n")

# --- Training Loop ---

def main():
    print(f"Initializing MLX Engine (AEDT Aware)...")
    
    # Data Loading (Mmap)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {DATA_PATH}")
    
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    print(f"Corpus Mapped. Size: {len(data)/1e6:.2f}M tokens.")
    
    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    # Model
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    mx.eval(model.parameters())
    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model Params: {params/1e6:.2f}M")
    
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY)
    
    def loss_fn(model, x, y):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses)
    
    state = [model.state, optimizer.state, mx.random.state]
    
    @mx.compile
    def train_step(x, y):
        loss_and_grads = mx.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grads(model, x, y)
        optimizer.update(model, grads)
        return loss

    # Loop State
    step = 0
    epoch = 0
    tokens_processed = 0
    last_coo_time = time.time()
    batch_size = STEALTH_BATCH_SIZE # Start conservative
    
    iter_start = time.time()
    
    print("Starting Infinite Training Loop...")
    
    try:
        while True:
            # 1. Governor Check
            target_bs, sleep_time, mode = get_governor_state(batch_size)
            batch_size = target_bs # Apply target
            
            # 2. Batch Construction
            ix = np.random.randint(0, len(data) - CONTEXT_LENGTH, batch_size)
            x_np = np.stack([data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int32)
            y_np = np.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int32)
            
            x = mx.array(x_np)
            y = mx.array(y_np)
            
            # 3. Step
            loss = train_step(x, y)
            mx.eval(state) # Sync
            
            # 4. Metrics & Logging
            dt = time.time() - iter_start
            iter_start = time.time()
            tokens_processed += batch_size * CONTEXT_LENGTH
            
            if step % 20 == 0:
                mem_gb = mx.metal.get_active_memory() / 1024**3
                tps = (batch_size * CONTEXT_LENGTH) / dt
                print(f"[Step {step}] {mode} | BS:{batch_size} | Loss:{loss.item():.3f} | Mem:{mem_gb:.1f}GB | {tps:.0f} tok/s")
                log_metrics(step, epoch, loss.item(), "N/A", mode, mem_gb, batch_size, tps)
                
            # 5. Cooing (Every 30 mins)
            if time.time() - last_coo_time > 1800: # 1800s = 30m
                generate_cooing(model, sp)
                last_coo_time = time.time()
            
            # 6. Checkpointing (End of "Epoch" approximation? Or strictly by steps?)
            # User said: "at the end of every Epoch".
            # Epoch = len(data) / (batch_size * ctx) steps.
            # Variable batch size makes exact epoch hard. Let's approx based on tokens seen.
            if tokens_processed >= len(data):
                epoch += 1
                tokens_processed = 0
                ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}.safetensors"
                print(f"Saving Epoch {epoch} checkpoint to {ckpt_path}...")
                model.save_weights(str(ckpt_path))
                
            # 7. Sleep (Stealth)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            step += 1
            
    except KeyboardInterrupt:
        print("\nPaused.")
        model.save_weights(str(CHECKPOINT_DIR / "interrupt_save.safetensors"))

if __name__ == "__main__":
    main()
