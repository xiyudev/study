#!/bin/zsh -ex
# Set up the environment for running manimgl: https://github.com/3b1b/manim.
#
# This script:
# - Installs the necessary system packages.
# - Creates a virtual environment (if not already created).
# - Installs the required Python packages (if not already installed).

brew install ffmpeg mactex pkg-config

# Create a virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

if [ ! -d "manimgl_clone" ]; then
    git clone https://github.com/3b1b/manim.git manimgl_clone
fi

# Activate the virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate

    pip install -e manimgl_clone
else
    echo "A virtual environment is already activated."
fi
