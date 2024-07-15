#!/bin/bash -ex
# Set up the environment for running manim.
# This script:
# - Installs the necessary system packages.
# - Creates a virtual environment (if not already created).
# - Sets up the environment variables.
# - Sets up the aliases.
# - Sets up the manim.cfg file.

# Update and upgrade the system
sudo apt update
# sudo apt upgrade -y

# Install dependencies
sudo apt install -y ffmpeg sox texlive-full libcairo2-dev libpango1.0-dev

# Create a virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
