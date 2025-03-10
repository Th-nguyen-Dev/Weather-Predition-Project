#!/bin/bash

# Directory containing the files
DIR="processed-final-data"
# Directory to save the downloaded files
SAVE_DIR="./location-data"

# Create the save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Loop through each file in the directory
for FILE in "$DIR"/*; do
    # Extract the file name from the path
    FILENAME=$(basename "$FILE")
    
    # Execute the wget command and save the file in the save directory
    wget -O "$SAVE_DIR/$FILENAME" https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/"$FILENAME"
done