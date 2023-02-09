#!/bin/bash

# Installing libraries
echo "Installing required packages"
pip3 install -r /content/requirements.txt
echo "Done!"

# Fetch needed files
echo "Cloning from GitHub"
git clone https://github.com/jessevig/bertviz.git
echo "Done!"
