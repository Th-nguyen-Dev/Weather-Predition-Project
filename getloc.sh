#!/bin/bash

# Directory containing the files
DIR="draft-final-data"

# Loop through each file in the directory
for FILE in "$DIR"/*; do
    # Extract the file name from the path
    FILENAME=$(basename "$FILE")
    
    # Execute the wget command
    wget -O "$FILENAME" https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/"$FILENAME"
done