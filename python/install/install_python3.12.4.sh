#!/bin/bash -ex
# This script installs Python 3.12.4 into a Ubuntu System (or WSL).

# Update and upgrade the system
sudo apt update
# sudo apt upgrade -y

# Install dependencies
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

# Create a temporary directory
mkdir /tmp/python3.12.4
cd /tmp/python3.12.4

# Download Python 3.12.4
wget https://www.python.org/ftp/python/3.12.4/Python-3.12.4.tgz

# Extract the downloaded file
tar -xf Python-3.12.4.tgz

# Go to the extracted directory
cd Python-3.12.4

# Configure the installation
./configure --enable-optimizations

# Build and install Python
make -j 4
sudo make altinstall
