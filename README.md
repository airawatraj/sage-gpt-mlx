# 🕉️ Sovereign Ancient General Intelligence (SAGE-GPT)

> "To find the Sutra in the Signal."

**Sage-GPT-7.25M (Grokking Phase)** is a Decoder-only Transformer trained from scratch on 56.89M ultra-pure Sanskrit tokens (164.8M characters). Architected to induce grokking (delayed generalization) through high-overfitting regimes.

### Specification Engine
* **Architecture**: 4 Layers, 8 Attention Heads, 256 Embedding Dim, 256 Context Length.
* **Tokenizer**: SentencePiece Unigram (8k Vocab, byte_fallback active, NFKC Strict).
* **Training Engine**: Strict AEDT-governed MLX engine: FACTORY mode (Batch Size 128) for maximum throughput, and STEALTH mode (Batch 4 x 32 Grad Accumulation) to maintain a mathematically stable effective batch size of 128 while keeping the M1 Mac's 16GB RAM free for work.

---

## 🚦 Execution Pipeline

### Step 1 & 2: Data Purification & Tokenization
Transforms 14GB raw data into the 56.89M token pure corpus using NFKC normalization and the 8k Sutra Unigram tokenizer.
```bash
uv run python3 1-data/src/prepare_data_v3.py
```

### Step 3: Train with AEDT Governor
Initiates the AEDT-governed MLX training loop. Defaults to AUTO governor, or use --mode FACTORY to force maximum throughput.
```bash
uv run python3 3-training/src/train_engine_mlx.py --mode FACTORY
```

### Step 4: Generalisation Gap Monitor
Runs passively in a separate terminal to generate a dark-mode, log-scale plot of the grokking divergence (Train vs Validation Loss).
```bash
uv run python3 4-evaluation/generalisation_gap_monitor.py
```

### Step 5: Run the Ashtavakra Audit

```bash
uv run python3 4-evaluation/ashtavakra_audit.py
```

---

## 🛡️ Linguistic Guardrails (V4)
Our "Zero-Poison" policy ensures the model trains only on high-fidelity Sanskrit.

| Feature | Implementation | Goal |
| :--- | :--- | :--- |
| **Normalization** | **NFKC** | Prevents shattering of conjuncts/roots (No NFC). |
| **Noise Shield** | Disjoint Stopwords | 100% rejection of Hindi, Marathi, Pali, and Prakrit. |
| **Punctuation** | Danda-Aware | Protects sutras even with trailing '।' or '॥' markers. |
| **Vedic Safety** | Swara Protection | Preservation of Visarga (ः), Anusvara (ं), and Vedic accents. |

---

## ⚙️ Operational Modes (AEDT Aware)
SAGE-GPT adapts training intensity based on the time of day to protect M1 resources.

### ☀️ STEALTH Mode (09:00 - 18:00)
* **Batch Size**: 4 (with 32-step Grad Accumulation).
* **Effective BS**: 128 (Mathematically stable).
* **VRAM**: Optimized for background execution while working.

### 🌙 FACTORY Mode (18:00 - 09:00)
* **Batch Size**: 128 (Direct Metal acceleration).
* **Focus**: High-throughput stochastic exploration.

---
> 🕉️ Om Tat Sat (ॐ तत् सत्) - The Absolute is Truth
