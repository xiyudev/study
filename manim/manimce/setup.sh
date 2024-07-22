#!/bin/bash -ex
# Set up the environment for running manimce - the Community Version.
#
# This script:
# - Installs the necessary system packages.
# - Creates a virtual environment (if not already created).
# - Installs the required Python packages (if not already installed).

# Update and upgrade the system
sudo apt update
# sudo apt upgrade -y

# Install dependencies
# This list below is generated by copilot.
#sudo apt install -y build-essential python3-dev ffmpeg sox texlive-full libcairo2-dev libpango1.0-dev
# This list below is from the official website https://docs.manim.community/en/stable/installation/linux.html.
sudo apt install build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg

# To allow using OpenGL renderer in WSL.
sudo apt install xvfb -y

# Create a virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate

    # Install the required Python packages if not already installed
    pip install -r requirements.txt
else
    echo "A virtual environment is already activated."
fi