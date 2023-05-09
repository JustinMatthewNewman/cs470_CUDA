#!/bin/bash

# Ensure input file name argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 input.png"
  exit 1
fi

input_file="$1"
output_directory="test"

# Create the output directory if it does not exist
mkdir -p "$output_directory"


echo "Testing -s option"
# Loop through the range from 1 to 250 with -s option
for i in $(seq 0 250); do
  # Generate the output file name
  output_file="${output_directory}/frame_s_${i}.png"

  # Call the program with the current -s value
  ./serial -s "$i" "$input_file" "$output_file"
  ./par -s "$i" "$input_file" "$output_file"

  # Print progress
  echo "Generated frame ${i}/250: ${output_file}"
done

echo "All frames generated with -s option in ${output_directory}"

echo "Testing -b option"
# Loop through the range from 0 to 100 with -b option
for i in $(seq 0 100); do
  # Generate the output file name
  output_file="${output_directory}/frame_b_${i}.png"

  # Call the program with the current -b value
  ./serial -b "$i" "$input_file" "$output_file"
  ./par -b 15 15 "$i" "$input_file" "$output_file"

  # Print progress
  echo "Generated frame ${i}/100: ${output_file}"
done

echo "All frames generated with -b option in ${output_directory}"

echo "Testing -d option"
./serial -d "$input_file" "$output_file"
./par -d "$input_file" "$output_file"
echo "All frames generated with -d option in ${output_directory}"


echo "Testing -r option"
./serial -r "$input_file" "$output_file"
./serial -r "$input_file" "$output_file"
./serial -r "$input_file" "$output_file"
./serial -r "$input_file" "$output_file"

./par -r "$input_file" "$output_file"
./par -r "$input_file" "$output_file"
./par -r "$input_file" "$output_file"
./par -r "$input_file" "$output_file"


echo "All frames generated with -r option in ${output_directory}"

