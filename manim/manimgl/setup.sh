#!/bin/bash -ex
# Set up the environment for running manimgl: https://github.com/3b1b/manim.
#
# Instruction: https://3b1b.github.io/manim/getting_started/installation.html
#
# This script:
# - Installs the necessary system packages.
# - Creates a virtual environment (if not already created).
# - Installs the required Python packages (if not already installed).

sudo apt update
sudo apt install ffmpeg

# Create a virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

if [ ! -d "manimgl_clone" ]; then
    git clone https://github.com/3b1b/manim.git manimgl_clone
fi

if [ ! -d "manimgl_videos" ]; then
    git clone https://github.com/3b1b/videos.git manimgl_videos
fi

# Activate the virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate

    pip install -e manimgl_clone
else
    echo "A virtual environment is already activated."
fi
