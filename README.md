# 🕉️ Sovereign Ancient General Intelligence (SAGE-GPT)

> "To find the Sutra in the Signal."

**SAGE-GPT** is an **8.4M parameter** LLM intentionally constrained to act as a linguistic distillation engine for Sanskrit. By deliberately limiting capacity to **4 layers and 8 heads**, we force the model toward **grokking**—the phase transition where a neural network internalizes Paninian grammatical logic rather than merely memorizing patterns.

---

## 🏛️ The Factory Structure (Verified)
The environment has been modularized to support high-throughput processing resulting in a 164.8M Ultra-Pure Character library.

* **1-data/**: Tiered pipeline from `01-raw` to `05-scripts` (visuddhi_v4 + refine_corpus).
* **2-tokenizer/**: SentencePiece Unigram (8k Vocab, byte_fallback active).
* **3-training/**: MLX-optimized engine with a Work-Aware AEDT Governor.
* **6-logs/**: Unified dashboard for purification and training metrics.

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

## 🚀 Execution

1. **Purify 14GB Library**:
   `uv run python3 1-data/05-scripts/visuddhi_v4.py`

2. **Train with AEDT Governor**:
   `uv run python3 3-training/src/train_engine_mlx.py`

3. **Run the Ashtavakra Audit**:
   `uv run python3 4-evaluation/ashtavakra_audit.py`

---
> 🕉️ Om Tat Sat (ॐ तत् सत्) - The Absolute is Truth
