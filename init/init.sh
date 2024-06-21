#!/bin/bash

# Ensure the script is running with root privileges
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# Update package list
apt-get update

# Install software-properties-common for managing PPAs
apt-get install -y software-properties-common

# Add the deadsnakes PPA for newer Python versions
add-apt-repository -y ppa:deadsnakes/ppa

# Update package list again after adding PPA
apt-get update

# Install Python 3.11 and the venv module
apt-get install -y python3.11 python3.11-venv

# Create a directory for virtual environments if it doesn't exist
mkdir -p ~/mypyenvs

# Create virtual environment named 'price'
python3.11 -m venv ~/mypyenvs/price

echo "Virtual environment 'price' created."
echo "To activate the virtual environment, run: source ~/mypyenvs/price/bin/activate"

source ~/mypyenvs/price/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install pandas
pip install pyarrow
pip install pandas_ta
pip install darts
pip install lightning 


sudo apt install nvtop

