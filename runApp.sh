#!/bin/bash

# Initialize Conda for this non-interactive shell
if [ -f "/home/nikhil/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/home/nikhil/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found."
    exit 1
fi

# Initialize conda for the current shell
conda init bash
# Source bashrc to apply conda initialization
source ~/.bashrc

# Activate the conda environment
echo "Activating conda environment: chessEnv"
conda activate chessEnv

# Set NVIDIA OpenGL environment
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/nvidia
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia:$LD_LIBRARY_PATH
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LIBGL_ALWAYS_INDIRECT=0

# Navigate to the directory where this script resides
cd "$(dirname "$0")" || exit 1

# Launch the GUI
echo "Launching Chess Move Predictor..."
python3 webGUI.py