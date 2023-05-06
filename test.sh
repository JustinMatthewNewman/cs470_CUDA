#!/bin/bash

echo "==========Testing Deterministic Options for Correctness=========="
echo ""
./serial -d input.png output.png > /dev/null
./par -d 1 input.png output2.png > /dev/null
result=$(compare -channel all -metric mae output.png output2.png diff.png 2>&1)

if [ "$result" = "0 (0)" ]; then
    echo "desaturation test... passed"
else
    echo "desaturation test... failed"
fi

./serial -r input.png output.png > /dev/null
./par -r input.png output2.png > /dev/null
result=$(compare -channel all -metric mae output.png output2.png diff.png 2>&1)

if [ "$result" = "0 (0)" ]; then
    echo "Rotate test... passed"
else
    echo "Rotate test... failed"
fi

echo "==========Times and Images for Option/Threshold Combinations=========="
echo ""
bash time.sh input.png
