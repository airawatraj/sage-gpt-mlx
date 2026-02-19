# 🕉️ Sovereign Ancient General Intelligence (SAGE)

> "To find the Sutra in the Signal."

**SAGE** is an initiative to distill the structural logic of the Sanskrit language into a pure, parameter-constrained neural architecture. By deliberately limiting the model's capacity, we force it to abandon rote memorization and instead *grok* the underlying grammatical and semantic rules of the Vedas and Upanishads.

---

## The Grokking Regime

We are currently operating under **Path B**, a high-pressure training regime designed to induce phase transitions in learning.

### The "Pressure Cooker" Architecture
Unlike massive LLMs that "memorize the internet," SAGE works within a strict parameter budget to ensure every weight encodes true generalization.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **19.85M** | A "Nano-Grade" intelligence. |
| **Dimensions ($D$)** | **384** | Narrow semantic channel width. |
| **Layers ($L$)** | **6** | Shallow depth to force efficiency. |
| **Heads ($H$)** | **6** | Focused attention mechanism. |
| **Context ($C$)** | **256** | Short horizon dependency modeling. |
| **Vocabulary** | **12k** | Custom SentencePiece BPE (Sutra-optimized). |

### The Truth Test (Validation Strategy)
We maintain a strict separation of church and state in our data:
*   **Total Corpus**: ~17.87M Tokens (Vedic + Classical Sanskrit).
*   **Training Split (90%)**: The raw material for the forge.
*   **Validation Split (10%)**: A "held-out" truth set used solely to detect the **Grokking Gap** (the divergence between memorization and generalization).

---

## ⚙️ Operational Modes

SAGE lives on your local hardware, respecting the rhythm of your day. The training engine features a **Work-Aware Governor**:

### ☀️ Stealth Mode (Day)
*   **Batch Size**: 4
*   **Behavior**: Low-VRAM usage, high sleep intervals. Allows you to code, browse, and work alongside the training process without lag. "The Sage meditates silently."

### 🌙 Factory Mode (Night)
*   **Batch Size**: 128
*   **Behavior**: Maximum throughput, full GPU utilization. "The Forge awakens."

---

## 📊 Analytics & Visualization

Watch the phase transition happen in real-time.

### The Grok-Plotter
Launch the live monitoring dashboard to visualize the **Loss Landscape** (Log Scale):
```bash
uv run python3 4-output/scripts/plot_grokking.py
```
*Visualizes `Train_Loss` (Green) vs. `Val_Loss` (Orange) with auto-refresh.*

### Optimization Dynamics
*   **Scheduler**: Cosine Decay with Warmup.
*   **Warmup**: 0 $\to$ $3\text{e}-4$ over 2,000 steps.
*   **Decay**: Tapers to $3\text{e}-5$ to settle into the energetic minima.

---

## 🚀 Setup & Initialization

SAGE is built on **Apple MLX** for metal-accelerated performance on M-Series chips.

### 1. Environment Sync
Ensure you are using the precise dependencies:
```bash
uv pip install -r requirements.txt
```

### 2. Ignite the Engine
To begin the training loop (auto-resumes from latest checkpoint):
```bash
uv run python 3-training/src/train_engine_mlx.py
```

### 3. Converse with SAGE
To test the model's generation capabilities:
```bash
uv run python 3-training/src/inference_engine_mlx.py
```

### 3. Hot Swap mode
To force hot swap FACTORY (Full Throttle Nightly) mode or STEALTH (Day Work) mode:
```bash
echo "FACTORY" > MODE_OVERRIDE.txt
#OR
echo "STEALTH" > MODE_OVERRIDE.txt
uv run python 3-training/src/train_engine_mlx.py

# Give control back to Governer
rm MODE_OVERRIDE.txt
```

### 4. Run Ashtavakra Audit
Run this to check the model's current state:
```bash
uv run python 3-training/src/ashtavakra_audit.py
```
---

> 🕉️ Om Tat Sat (ॐ तत् सत्) - The Absolute is Truth