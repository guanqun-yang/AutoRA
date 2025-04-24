#!/bin/bash

# Exit on error
set -e

# Define paths
ZIP_PATH=arxiv.zip
EXTRACT_DIR=.

# Download from Kaggle (requires kaggle.json auth set up)
echo "Downloading arxiv.zip from Kaggle..."
curl -L -o "$ZIP_PATH" \
  https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv

# Create target directory if it doesn't exist
mkdir -p "$EXTRACT_DIR"

# Unzip the file
echo "Extracting arxiv.zip to $EXTRACT_DIR..."
unzip -o "$ZIP_PATH" -d "$EXTRACT_DIR"

# Optional: remove the zip file
echo "Cleaning up zip file..."
rm "$ZIP_PATH"

echo "Done."