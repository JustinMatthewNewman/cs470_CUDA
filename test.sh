#!/bin/bash

echo "==========Testing Deterministic Options for Correctness=========="
echo ""
./serial -d input.png output.png > /dev/null
./par -d input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "desaturation test... passed"
else
    echo "desaturation test... failed"
fi

./serial -g 10 input.png output.png > /dev/null
./par -g 10 input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Gaussian Blur test... passed"
else
    echo "Gaussian Blur test... failed"
fi

./serial -r input.png output.png > /dev/null
./par -r input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Rotate test... passed"
else
    echo "Rotate test... failed"
fi

echo "==========Times and Images for Option/Threshold Combinations=========="
echo ""
bash time.sh input.png
