# 🕉️ SAGE-GPT: Sanskrit Language Engine

A high-performance transformer model designed for Sanskrit text generation and spiritual text analysis. Optimized for Apple Silicon (MLX).

## 🚀 Operational Architecture

### 1. Environment & Setup
The engine utilizes \`uv\` for lightning-fast dependency management and environment isolation.
- **Python**: 3.13+
- **Framework**: MLX (Apple Silicon optimized)

### 2. The Training Engine
The core training logic is contained in \`3-training/src/train_engine_mlx.py\`. It features an automated power-management system:
- **Factory Mode (Night)**: Maximum throughput for rapid convergence.
- **Stealth Mode (Day)**: Throttled resource usage for simultaneous development work.

### 3. Monitoring & Analytics
Monitor the Sage's evolution through integrated logging:
- **Corpus Monitoring**: Tracks the expansion of binary data.
- **Real-time Metrics**: Validation loss and training history tracked in \`4-output/logs/\`.
- **Sample Generation**: Periodic Sanskrit clusters output to terminal for quality assessment.

## 🛡️ Data Integrity
This repository contains the **logic only**. All training corpora, binary weights, and environment artifacts are excluded to ensure a secure and lightweight repository structure.