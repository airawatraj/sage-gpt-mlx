import mlx.core as mx
import sys
from pathlib import Path

def main():
    # Define the path to the specified checkpoint
    checkpoint_path = Path("3-model/mlx/checkpoints/interrupt_save.safetensors")
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    weights = mx.load(str(checkpoint_path))
    
    print("\nL2 Norms of Attention and MLP Layer Weight Matrices:")
    print("-" * 60)
    
    # Track statistics
    attn_norms = []
    mlp_norms = []
    
    # Calculate and log L2 norm for targeted parameters
    for key, tensor in sorted(weights.items()):
        if "attn" in key or "mlp" in key:
            if "weight" in key:
                # Compute the L2 norm for the weight matrix
                l2_norm = mx.sqrt(mx.sum(mx.square(tensor))).item()
                print(f"{key:.<50} {l2_norm:.4f}")
                
                if "attn" in key:
                    attn_norms.append(l2_norm)
                elif "mlp" in key:
                    mlp_norms.append(l2_norm)

    print("-" * 60)
    if attn_norms:
        print(f"Average Attention Weight L2 Norm: {sum(attn_norms) / len(attn_norms):.4f}")
    if mlp_norms:
        print(f"Average MLP Weight L2 Norm:       {sum(mlp_norms) / len(mlp_norms):.4f}")

if __name__ == "__main__":
    main()
