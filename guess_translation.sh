#!/bin/bash

docker run -it --rm \
       -e XAUTHORITY=/home/mrd45/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v $PWD:/work \
       -u `id -u`:`id -g` \
       guess_translation python /guess_translation.py $*
