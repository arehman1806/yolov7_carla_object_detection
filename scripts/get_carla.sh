#!/bin/bash
# Download command: bash ./scripts/get_carla.sh

# Download/unzip images and labels
d='./' # destination directory
url=https://github.com/DanielHfnr/Carla-Object-Detection-Dataset/archive/refs/tags/
version='1.0'
f='v'$version'.zip'
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f
mv Carla-Object-Detection-Dataset-$version carla
rm -rf ./carla/labels
mv ./carla/labels_yolo_format ./carla/labels
echo 'Done'