#!/bin/bash

# Check for filename
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <image> [flags]"
    exit 1
fi

# Get the image filename
image_filename=${@: -1}

# Extract image dimensions using the ImageMagick 'identify' command
dims=$(identify "${image_filename}" | cut -f 3 -d ' ')
width=$(echo "${dims}" | cut -f 1 -d 'x')
height=$(echo "${dims}" | cut -f 2 -d 'x')

# Display help message if no dimensions are found
if [[ -z "${dims}" ]]; then
  help=true
fi

# Create a temporary file in PPM format
temp_file="converted_image.ppm"
convert "${image_filename}" -compress None PPM:"${temp_file}"

# Build the command line arguments for ./serial
args=("${temp_file}" "output_serial.ppm" "${width}" "${height}")
while [[ $# -gt 1 ]]; do
    key="$1"
    case ${key} in
        -g)
            args+=("-g" "$2")
            shift
            shift
            ;;
        -d)
            args+=("-d" "$2")
            shift
            shift
            ;;
        -r)
            args+=("-r")
            shift
            ;;
        -b)
            args+=("-b" "$2")
            shift
            shift
            ;;
        -s)
            args+=("-s" "$2")
            shift
            shift
            ;;
        *)
            echo "ERROR: Invalid flag ${key}"
            exit 1
            ;;
    esac
done

# Run serial program if it is present
if [ -e "serial" ]; then
    echo -n "Serial:       "
    ./serial "${args[@]}"
    convert output_serial.ppm PNG:output_serial.png
    rm output_serial.ppm
else
    echo "ERROR: Serial program not found"
fi

# Remove the temporary file
rm "${temp_file}"
