#!/bin/bash

echo "==========Testing Deterministic Options for Correctness=========="
echo ""
./serial -d input.png output.png > /dev/null
./par -d input.png output2.png > /dev/null
result=$(compare -channel all -metric mae output.png output2.png diff.png 2>&1)

if [ "$result" = "0 (0)" ]; then
    echo "desaturation test... passed"
else
    echo "desaturation test... failed ($result)"
fi

./serial -r input.png output.png > /dev/null
./par -r input.png output2.png > /dev/null
result=$(compare -channel all -metric mae output.png output2.png diff.png 2>&1)

if [ "$result" = "0 (0)" ]; then
    echo "Rotate test... passed"
else
    echo "Rotate test... failed ($result)"
fi

./serial -s 150 input.png output.png > /dev/null
./par -s 150 input.png output2.png > /dev/null
result=$(compare -channel all -metric mae output.png output2.png diff.png 2>&1)

if [ "$result" = "0 (0)" ]; then
    echo "Sorting test... passed"
else
    echo "Sorting test... failed ($result)"
fi

./serial -b 75 input.png output.png > /dev/null
./par -b 15 15 75 input.png output2.png > /dev/null
result=$(compare -channel all -metric mae output.png output2.png diff.png 2>&1)

if [ "$result" = "0 (0)" ]; then
    echo "Background Removal test... passed"
else
    echo "Background Removal test... failed ($result)"
fi
echo "==========Times and Images for Option/Threshold Combinations=========="
echo ""
bash time.sh input.png
