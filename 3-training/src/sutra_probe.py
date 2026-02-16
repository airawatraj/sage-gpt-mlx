import mlx.core as mx
import sys
from pathlib import Path

# Path to checkpoint
checkpoint_path = Path("3-model/mlx/checkpoints/epoch_14.safetensors")

try:
    weights = mx.load(str(checkpoint_path))
    print(f"Loaded weights from {checkpoint_path}")
    print(f"Total keys: {len(weights)}")
    
    # 1. Check Embedding Weight Shape
    if "embedding.weight" in weights:
        print(f"embedding.weight shape: {weights['embedding.weight'].shape}")
    elif "token_embedding.weight" in weights:
        print(f"token_embedding.weight shape: {weights['token_embedding.weight'].shape}")
    else:
        print("No embedding weight found!")

    # 2. Check Layer Count
    max_layer = -1
    for key in weights.keys():
        if "blocks." in key:
            try:
                layer_num = int(key.split(".")[1])
                max_layer = max(max_layer, layer_num)
            except:
                pass
    print(f"Max Block Index found: {max_layer} (Implies {max_layer+1} layers)")

    # 3. Check for Position Embeddings
    if "position_embedding.weight" in weights:
        print("Found position_embedding.weight (Learned Pos Emb)")
    else:
        print("No position_embedding.weight found (Likely RoPE)")

    # 4. Check GELU/ReLU? Hard to tell from weights, but look for specific names if any?
    # Usually activation doesn't have weights.

except Exception as e:
    print(f"Error loading checkpoint: {e}")
