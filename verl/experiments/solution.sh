#!/bin/bash
# Update package list
sudo apt-get update

# Install python3-pip if not already installed
sudo apt-get install -y python3-pip

# Install the requests library
pip install requests

# Run a simple Python script to test the installation
python3 -c "import requests; print(requests.get('https://example.com').status_code)"
