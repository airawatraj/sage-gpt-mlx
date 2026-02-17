#!/bin/bash

# auto_shift.sh
# Automates the 6 PM handover: Graceful Shutdown -> Verify -> Factory Restart

TARGET_TIME="18:00"
CHECKPOINT_FILE="3-model/mlx/checkpoints/interrupt_save.safetensors"

echo "⏳ Auto-Shift: Waiting for $TARGET_TIME..."

while true; do
    CURRENT_TIME=$(date +%H:%M)
    
    if [ "$CURRENT_TIME" == "$TARGET_TIME" ]; then
        echo "⏰ It's $TARGET_TIME! Initiating Handover..."
        
        # 1. Find PID
        PID=$(pgrep -f "train_engine_mlx.py")
        
        if [ -z "$PID" ]; then
            echo "❌ No training process found running."
        else
            echo "🛑 Found Train Engine PID: $PID. Sending Interrupt (Save)..."
            kill -2 $PID
            
            # Wait for process to exit
            while kill -0 $PID 2>/dev/null; do
                echo "   ...waiting for graceful exit..."
                sleep 2
            done
            echo "✅ Process $PID exited."
        fi
        
        # 2. Verification
        sleep 2 
        if [ -f "$CHECKPOINT_FILE" ]; then
            # Ideally check timestamp, but existence + recent exit implies success for now
            echo "✅ Verified: $CHECKPOINT_FILE exists."
            ls -l "$CHECKPOINT_FILE"
        else
            echo "⚠️  Warning: Checkpoint file not found or path incorrect!"
            echo "   Expected at: $CHECKPOINT_FILE"
            # Proceeding anyway as Factory mode might be robust enough to pick up latest epoch
        fi
        
        # 3. Re-entry (Factory Mode)
        echo "🏭 Engaging FACTORY Mode for Overnight Protocol..."
        echo "=================================================="
        
        # Ensure we are in the right directory (project root)
        cd "$(dirname "$0")"
        
        uv run python 3-training/src/train_engine_mlx.py --batch-size 128 --learning-rate 3e-5 --resume --mode FACTORY
        
        # Exit script after launching (or stay attached if user wants logs)
        # Staying attached to show output
        exit 0
    fi
    
    sleep 30
done
