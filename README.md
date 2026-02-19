# 🕉️ Sovereign Ancient General Intelligence (SAGE)

> "To find the Sutra in the Signal."

**SAGE** is an initiative to distill the structural logic of the Sanskrit language into a pure, parameter-constrained neural architecture. By deliberately limiting capacity, we force the model to move past rote memorization and toward *grokking*—the phase transition where a model internalizes grammatical and semantic rules.

---

## The Grokking Regime

We are operating under a high-pressure regime designed to induce delayed generalization through extreme overfitting on a high-quality Sanskrit corpus.

### The "Pressure Cooker" Architecture (Verified)
Built for the 16GB M1 Pro, SAGE uses a narrow but deep configuration to maximize the density of representation.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | **16.78M** | Optimized for "Nano-Grade" intelligence. |
| **Dimensions ($D$)** | **384** | Narrow semantic channel. |
| **Layers ($L$)** | **6** | Transformer blocks for hierarchy. |
| **Heads ($H$)** | **6** | Multi-head attention (64 dim/head). |
| **Context ($C$)** | **512** | Local dependency horizon. |
| **Vocabulary** | **8,000** | SentencePiece Unigram (NFKC Normalized). |

### The Truth Test
* **Total Corpus**: ~80.93M Tokens (Vedic, Classical, & Shastra).
* **Tokenization**: NFKC normalization ensures Sanskrit roots are preserved.
* **Split**: 90% Train / 10% Val (used to monitor the **Grokking Gap**).

---

## 🐚 The 8 Bends (Ashtavakra Audit)
To measure grokking, we use a specialized hybrid diagnostic that tests the model against the "8 Bends" of Sanskrit linguistic logic.

| Bend | Test | Goal |
| :--- | :--- | :--- |
| **1** | Phonetic Stability | Checks for stuttering in "ॐ" flow. |
| **2** | Invocation | Completion of "असतो मा" > "सद्गमय". |
| **3** | Case Inflection | Validates Visarga probability ($P(ः) > 0.4$) for "राम". |
| **4** | Name Synthesis | Tests Guna Sandhi (नर + इन्द्र = नरेन्द्र). |
| **5** | Concept Flow | Mundaka Upanishad flow (यथा नद्यः > समुद्रे). |
| **6** | Verse Sequence | Checks Isha Upanishad continuity. |
| **7** | Orthography | Validates retroflexion clusters (कृष् > ण). |
| **8** | The Atman Test | Mahavakya completion (तत्त्वमसि > श्वेतकेतो). |

### Run the Audit
```bash
uv run python3 3-training/src/ashtavakra_audit.py
```

---

## ⚙️ Operational Modes (AEDT Aware)

SAGE utilizes a **Work-Aware Governor** that adapts training intensity based on the time of day.

### ☀️ STEALTH Mode (09:00 - 18:00)
* **Batch Size**: 4 (Micro) with 32-step Gradient Accumulation.
* **Effective BS**: 128 (Mathematically stable).
* **VRAM**: ~0.4GB. Designed for silent background meditation while you work.

### 🌙 FACTORY Mode (18:00 - 09:00)
* **Batch Size**: 128 (Direct).
* **Throughput**: Maximum Metal acceleration.
* **Focus**: Stochastic exploration of the loss landscape.

---

## 🚀 Execution & Interaction

### 1. Environment Setup
```bash
uv pip install -r requirements.txt
```

### 2. The Training Loop
```bash
uv run python3 3-training/src/train_engine_mlx.py
```

### 3. Dual-Mode Inference
```bash
uv run python3 3-training/src/inference_engine_mlx_v2.py
```

---

## 📊 Monitoring
```bash
uv run python3 4-output/scripts/plot_grokking.py
```

---
> 🕉️ Om Tat Sat (ॐ तत् सत्) - The Absolute is Truth
