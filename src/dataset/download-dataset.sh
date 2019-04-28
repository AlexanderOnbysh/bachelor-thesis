#!/bin/bash
echo "Download dataset"
declare -a dataset=("http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partaa" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partab" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partac" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partad" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partae" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partaf" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partag" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_trainval.zip" \
                    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_test_v0.4.zip" \
                    )

for url in "${dataset[@]}"
do
    wget --user $USER --password $PASSWORD -P $1 $url
done

echo "cat $1/lrs3_pretrain_part* > $1/lrs3_pretrain.zip"