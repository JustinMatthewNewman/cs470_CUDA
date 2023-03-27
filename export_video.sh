#!/bin/bash


# Ensure input and output file name arguments are provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 input.png"
  exit 1
fi

input_file="$1"

bash encode_video/generate_frames.sh "$input_file" "encode_video/frames"


ffmpeg -r 24 -i encode_video/frames/frame_%d.png -c:v libx264 -pix_fmt yuv444p -crf 23 encode_video/output.mp4
