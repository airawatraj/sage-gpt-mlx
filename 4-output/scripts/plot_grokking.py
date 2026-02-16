
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter
import sys
import os
from pathlib import Path

# --- Configuration ---
CSV_PATH = Path("4-output/logs/training_history.csv")
REFRESH_INTERVAL_MS = 60000 

# --- Theme Setup ---
plt.style.use('dark_background')
SAGE_GREEN = '#9DC183'
SAGE_ORANGE = '#E8A87C'
SAGE_RED = '#C38D9E'
SAGE_BLUE = '#41B3A3'

def update_plot(frame):
    if not CSV_PATH.exists():
        print(f"Waiting for {CSV_PATH}...")
        return

    try:
        # Read CSV with manual headers (since file might be raw)
        # Column 0: Timestamp, 1: Step, 2: Epoch, 3: TrLoss, 4: ValLoss, 5: Mode, 6: Mem, 7: BS, 8: TPS, 9: LR
        col_names = ['Timestamp', 'Step', 'Epoch', 'Train_Loss', 'Val_Loss', 'Mode', 'Memory_GB', 'Batch_Size', 'Tokens_Per_Sec', 'LR']
        df = pd.read_csv(CSV_PATH, header=None, names=col_names)
        
        # Ensure we have data
        if df.empty:
            return

        # Filter out potential header rows (where 'Step' column contains the string 'Step')
        df = df[pd.to_numeric(df['Step'], errors='coerce').notnull()]
            
        # Clean Data (handle potential N/A strings if any legacy rows exist)
        # In new strict mode, Val_Loss should be numeric or castable
        df['Val_Loss'] = pd.to_numeric(df['Val_Loss'], errors='coerce')
        df['Train_Loss'] = pd.to_numeric(df['Train_Loss'], errors='coerce')
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        
        # Drop rows with NaN steps/losses just in case
        df = df.dropna(subset=['Step', 'Train_Loss'])

        # Clear current axes
        plt.cla()
        
        # Plot Curves
        plt.plot(df['Step'], df['Train_Loss'], label='Train Loss', color=SAGE_GREEN, alpha=0.8, linewidth=1.5)
        
        # Handle Val_Loss potentially being empty/NaN in early steps
        if 'Val_Loss' in df.columns:
            val_df = df.dropna(subset=['Val_Loss'])
            if not val_df.empty:
               plt.plot(val_df['Step'], val_df['Val_Loss'], label='Val Loss', color=SAGE_ORANGE, alpha=0.9, linewidth=2.0)

        
        # Log Scale Y
        plt.yscale('log')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        
        # Annotations
        # Warmup End (Step 2000)
        if df['Step'].max() >= 2000:
            plt.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, label='Warmup End')
        
        # Labels & Title
        plt.title(f"SAGE-GPT Learning Curves (Grokking Regime)\nLatest Tr: {df['Train_Loss'].iloc[-1]:.4f} | Val: {df['Val_Loss'].iloc[-1]:.4f}", fontsize=14, color='white')
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Loss (Log Scale)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="-", alpha=0.1)
        
        # Adjust layout
        plt.tight_layout()
        
    except Exception as e:
        print(f"Error updating plot: {e}")

def main():
    print(f"Starting Grokking Monitor on {CSV_PATH}...")
    fig = plt.figure(figsize=(12, 6))
    ani = animation.FuncAnimation(fig, update_plot, interval=REFRESH_INTERVAL_MS, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    current_dir = Path.cwd()
    if not (current_dir / "4-output").exists():
        print("Error: Run this script from the project root (e.g., python 4-output/scripts/plot_grokking.py)")
        sys.exit(1)
    main()
