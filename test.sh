#!/bin/bash

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

./serial -b 10 input.png output.png > /dev/null
./par -b 10 input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Background Removal test... passed"
else
    echo "Background Removal test... failed"
fi

./serial -a 10 input.png output.png > /dev/null
./par -a 10 input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Background Removal Averaging test... passed"
else
    echo "Background Removal Averaging test... failed"
fi

./serial -f 10 input.png output.png > /dev/null
./par -f 10 input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Foreground Removal test... passed"
else
    echo "Foreground Removal test... failed"
fi

./serial -t 10 15 15 input.png output.png > /dev/null
./par -t 10 15 15 input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Target Removal test... passed"
else
    echo "Target Removal test... failed"
fi

./serial -s 10 input.png output.png > /dev/null
./par -s 10 input.png output2.png > /dev/null
compare -channel all -metric mae output.png output2.png diff.png > result.txt 2>&1
grep '0 (0)' result.txt > /dev/null

if [ $? == 0 ]; then
    echo "Sorting test... passed"
else
    echo "Sorting test... failed"
fi
