#!/bin/bash

docker run --gpus 1 --net=host -it --rm \
       -e XAUTHORITY="$HOME"/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v $PWD:/work \
       -v "$HOME":"$HOME":ro \
       -u `id -u`:`id -g` \
       deep_sort_yolov3_count count.sh $*
