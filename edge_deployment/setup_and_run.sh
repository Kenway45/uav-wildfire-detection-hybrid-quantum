#!/bin/bash
# UAV Edge Pipeline Setup for Raspberry Pi VM
# Run this script after mounting the shared folder

echo "Setting up UAV Fire Detection on Raspberry Pi..."

# Step 1: Install dependencies
pip3 install onnxruntime opencv-python-headless numpy pillow

# Step 2: Create output folder
mkdir -p /mnt/qpca/Fire_Alerts

echo "Setup complete! Now running the fire detector..."

# Step 3: Run the detector directly from the shared folder
python3 /mnt/qpca/step3_pi.py
