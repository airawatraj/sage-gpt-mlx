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
# Architecture (Grokking Regime: ~15M params)
VOCAB_SIZE = 12000
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
CONTEXT_LENGTH = 256

# Training
WEIGHT_DECAY = 0.1
LEARNING_RATE_MAX = 3e-4
LEARNING_RATE_MIN = 3e-5
WARMUP_STEPS = 2000
LR_DECAY_STEPS = 100000

# Governor (AEDT UTC+11)
AEDT_OFFSET = timezone(timedelta(hours=11))
STEALTH_BATCH_SIZE = 4
FACTORY_BATCH_SIZE = 128
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
MODE_OVERRIDE_FILE = project_root / "MODE_OVERRIDE.txt"

# --- Logging Setup ---
# --- Logging Setup ---
file_exists = list(LOG_FILE.exists() for _ in [0]) # Hacky check for header
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        writer.writerow(["Timestamp", "Step", "Epoch", "Train_Loss", "Val_Loss", "Mode", "Memory_GB", "Batch_Size", "Tokens_Per_Sec", "LR"])

def log_metrics(step, epoch, train_loss, val_loss, mode, mem_gb, batch_size, tps, lr):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss
        writer.writerow([datetime.now(AEDT_OFFSET).isoformat(), step, epoch, f"{train_loss:.4f}", val_str, mode, f"{mem_gb:.2f}", batch_size, f"{tps:.2f}", f"{lr:.2e}"])

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

def get_lr(it):
    # 1) Linear Warmup
    if it < WARMUP_STEPS:
        return LEARNING_RATE_MAX * (it + 1) / WARMUP_STEPS
    # 2) Constant Min Learning Rate after Decay Info
    if it > LR_DECAY_STEPS:
        return LEARNING_RATE_MIN
    # 3) Cosine Decay
    decay_ratio = (it - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE_MIN + coeff * (LEARNING_RATE_MAX - LEARNING_RATE_MIN)

def get_governor_state(current_batch_size, override_status=None):
    """
    Returns (target_batch_size, sleep_time, mode_name) based on AEDT time and Memory.
    override_status: If set to "FACTORY" or "STEALTH", forces that mode.
    """
    # 0. Manual Override
    if override_status == "FACTORY":
        return FACTORY_BATCH_SIZE, 0.0, "FACTORY (OVERRIDE)"
    elif override_status == "STEALTH":
        return STEALTH_BATCH_SIZE, STEALTH_SLEEP, "STEALTH (OVERRIDE)"

    # 1. Memory Safety Override
    active_mem = mx.get_active_memory()
    if active_mem > MEMORY_LIMIT_BYTES:
        print(f"[SAFETY] Memory {active_mem/1024**3:.2f}GB > Limit. Dropping Batch Size.")
        new_bs = max(1, current_batch_size // 2)
        mx.clear_cache()
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

def get_last_step():
    """Reads the last step from the training log CSV."""
    if not LOG_FILE.exists():
        print(f"[RESUME] Log file not found at {LOG_FILE}")
        return 0
    try:
        # Use 'rb' to efficiently seek to end, but for simplicity/robustness with CSV module, read text
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            if not lines:
                print("[RESUME] Log file is empty.")
                return 0
            
            # Get last non-empty line
            last_line = lines[-1].strip()
            if not last_line:
                # Try going back
                last_line = lines[-2].strip() if len(lines) > 1 else ""

            if not last_line:
                 print("[RESUME] No valid lines found in log.")
                 return 0
            
            # Parse CSV line manually or with reader on just that line
            row = next(csv.reader([last_line]))
            # Header: Timestamp,Step,Epoch,...
            # Step is index 1
            step_val = int(row[1])
            print(f"[RESUME] Found last step in log: {step_val}")
            return step_val

    except Exception as e:
        print(f"[WARN] Failed to read last step from log: {e}")
        return 0

def get_latest_checkpoint():
    """Finds the latest checkpoint (interrupt_save or highest epoch)."""
    # 1. Check for interrupt_save (highest priority for immediate resume)
    interrupt_ckpt = CHECKPOINT_DIR / "interrupt_save.safetensors"
    if interrupt_ckpt.exists():
        return interrupt_ckpt
    
    # 2. Check for numbered epochs
    checkpoints = list(CHECKPOINT_DIR.glob("epoch_*.safetensors"))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    try:
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
        return latest
    except ValueError:
        return None

# --- Training Loop ---

def main():
    print(f"Initializing MLX Engine (AEDT Aware)...")
    
    # Data Loading (Mmap & Split)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {DATA_PATH}")
    
    data_map = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    split_idx = int(len(data_map) * 0.9)
    train_data = data_map[:split_idx]
    val_data = data_map[split_idx:]
    
    print(f"Corpus Mapped. Total: {len(data_map)/1e6:.2f}M tokens.")
    print(f"Split: Train={len(train_data)/1e6:.2f}M, Val={len(val_data)/1e6:.2f}M")
    
    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    # Model
    model = TransformerLM(VOCAB_SIZE, N_LAYER, N_EMBD, N_HEAD)
    mx.eval(model.parameters())
    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model Params: {params/1e6:.2f}M")
    
    # --- Resume Logic ---
    start_step = 0
    
    # 1. Load Weights
    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt:
        try:
            print(f"Loading weights from {latest_ckpt.name}...")
            model.load_weights(str(latest_ckpt))
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}")
    
    # 2. Restore Step Count (STRICT STEP LOADING)
    last_step_from_csv = get_last_step()
    if last_step_from_csv > 0:
        start_step = last_step_from_csv
        print(f"Resumed Step Count: {start_step}")
    else:
        print("[RESUME] Starting from Step 0 (No history found).")
    
    # Optimizer (LR updated in loop)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE_MAX, weight_decay=WEIGHT_DECAY)
    
    def loss_fn(model, x, y):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses)
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Val Function
    def estimate_loss():
        losses = []
        # 10 random batches
        for _ in range(10):
            ix = np.random.randint(0, len(val_data) - CONTEXT_LENGTH, STEALTH_BATCH_SIZE) # Use small batch for speed
            x_np = np.stack([val_data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int32)
            y_np = np.stack([val_data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int32)
            x = mx.array(x_np)
            y = mx.array(y_np)
            logits = model(x)
            l = nn.losses.cross_entropy(logits, y)
            losses.append(mx.mean(l).item())
        return sum(losses) / len(losses)

    # Loop State
    step = start_step # Explicit State Override from CSV
    print(f"[DEBUG] Loop initialized with Step: {step}")
    epoch = 0 # TODO: Could approximate epoch from step if needed
    tokens_processed = 0
    last_coo_time = time.time()
    
    # --- Initial Governor Check ---
    override_status = None
    if MODE_OVERRIDE_FILE.exists():
        try:
            content = MODE_OVERRIDE_FILE.read_text().strip()
            if content in ["FACTORY", "STEALTH"]:
                override_status = content
        except Exception:
            pass

    # Set initial batch size based on Governor immediately
    batch_size, sleep_time, mode = get_governor_state(STEALTH_BATCH_SIZE, override_status)
    print(f"Resuming from Step {step} in {mode} Mode (BS={batch_size})")
    
    iter_start = time.time()
    
    print("Starting Infinite Training Loop...")
    
    try:
        while True:
            # 0. Check for Override Signal (Every 500 steps, synced with Validation)
            if step % 500 == 0:
                new_override = None
                if MODE_OVERRIDE_FILE.exists():
                    try:
                        content = MODE_OVERRIDE_FILE.read_text().strip()
                        if content in ["FACTORY", "STEALTH"]:
                            new_override = content
                    except Exception as e:
                        print(f"[WARN] Failed to read override file: {e}")
                
                # Check for State Change
                if new_override != override_status:
                    print(f"\n⚠️ SOVEREIGN OVERRIDE: TRANSITIONING TO [{new_override if new_override else 'AUTO'}] ⚠️")
                    
                    # 1. Save Interruption State
                    msg = f"Transitioning to {new_override}" if new_override else "Returning to Auto Governor"
                    print(f"[Governor] {msg}. Saving safety checkpoint...")
                    model.save_weights(str(CHECKPOINT_DIR / "interrupt_save.safetensors"))
                    
                    # 2. Clear Cache for Re-allocation
                    mx.clear_cache()
                    print("[Governor] Metal Cache Cleared.")
                    
                    override_status = new_override

            # 1. Governor Check
            target_bs, sleep_time, mode = get_governor_state(batch_size, override_status)
            batch_size = target_bs # Apply target
            
            # 2. Update LR
            lr = get_lr(step)
            optimizer.learning_rate = lr

            # 3. Batch Construction (Train split)
            ix = np.random.randint(0, len(train_data) - CONTEXT_LENGTH, batch_size)
            x_np = np.stack([train_data[i:i+CONTEXT_LENGTH] for i in ix]).astype(np.int32)
            y_np = np.stack([train_data[i+1:i+CONTEXT_LENGTH+1] for i in ix]).astype(np.int32)
            
            x = mx.array(x_np)
            y = mx.array(y_np)
            
            # 4. Step
            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state) # Sync
            
            # 5. Metrics & Logging
            dt = time.time() - iter_start
            iter_start = time.time()
            tokens_processed += batch_size * CONTEXT_LENGTH
            
            if step % 500 == 0:
                val_loss = estimate_loss()
                mem_gb = mx.get_active_memory() / 1024**3
                tps = (batch_size * CONTEXT_LENGTH) / dt
                print(f"[Step {step}] {mode} | BS:{batch_size} | TrLoss:{loss.item():.4f} | ValLoss:{val_loss:.4f} | LR:{lr:.2e} | Mem:{mem_gb:.1f}GB")
                log_metrics(step, epoch, loss.item(), val_loss, mode, mem_gb, batch_size, tps, lr)
            elif step % 20 == 0:
                 # Fast log
                mem_gb = mx.get_active_memory() / 1024**3
                tps = (batch_size * CONTEXT_LENGTH) / dt
                print(f"[Step {step}] {mode} | BS:{batch_size} | Loss:{loss.item():.4f} | LR:{lr:.2e} | {tps:.0f} tok/s")

                
            # 5. Cooing (Every 30 mins)
            if time.time() - last_coo_time > 1800: # 1800s = 30m
                generate_cooing(model, sp)
                last_coo_time = time.time()
            
            # 6. Checkpointing (End of "Epoch" approximation? Or strictly by steps?)
            # User said: "at the end of every Epoch".
            # Epoch = len(data) / (batch_size * ctx) steps.
            # Variable batch size makes exact epoch hard. Let's approx based on tokens seen.
            if tokens_processed >= len(train_data):
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
