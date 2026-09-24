#!/bin/bash
# Auto-Chaining Watcher: Seamlessly hand off from 0.5B to 1.5B training on Tesla V100.
set -e

TARGET_PID=368
echo "[*] Auto-Chaining Watcher Active. Monitoring 0.5B training (PID: $TARGET_PID)..."

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 30
done

echo "[*] 0.5B Training (PID: $TARGET_PID) has completed!"
sleep 15

WEIGHT_05B="/workspace/code/output/s1_model_v6/s1_decision_weights.pt"
if [ -f "$WEIGHT_05B" ]; then
    echo "[+] Confirmed: 0.5B baseline weights generated successfully."
else
    echo "[!] Note: PID exited, proceeding to launch 1.5B flagship."
fi

echo "[*] Launching Phase 2: Aegis-S1 1.5B Flagship Training on Tesla V100..."
cd /workspace/code
mkdir -p /workspace/code/output/s1_model_1.5b

nohup /workspace/venv/bin/python -u /workspace/code/scripts/train_s1.py \
    --base_model /workspace/models/base/Qwen2.5-1.5B-Instruct \
    --train_file /workspace/code/data/disjoint_v6/train_v6_disjoint.json \
    --calib_file /workspace/code/data/disjoint_v6/calib_v6_disjoint.json \
    --test_file /workspace/code/data/disjoint_v6/test_v6_disjoint.json \
    --output_dir /workspace/code/output/s1_model_1.5b \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr_head 2e-4 \
    --lr_lora 1e-4 \
    > /workspace/code/output/training_1.5b.log 2>&1 &

NEW_PID=$!
echo "[+] 1.5B Training launched successfully with PID: $NEW_PID"
echo "[+] Live log: /workspace/code/output/training_1.5b.log"
