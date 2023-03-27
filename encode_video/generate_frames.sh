#!/bin/bash

# Ensure input and output file name arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input.png output_directory"
  exit 1
fi

input_file="$1"
output_directory="$2"

# Create the output directory if it does not exist
mkdir -p "$output_directory"

# Loop through the range from 1 to 250
for i in $(seq 1 250); do
  # Generate the output file name
  output_file="${output_directory}/frame_${i}.png"

  # Call the program with the current -s value
  ./serial -s "$i" "$input_file" "$output_file"

  # Print progress
  echo "Generated frame ${i}/250: ${output_file}"
done

echo "All frames generated in ${output_directory}"
