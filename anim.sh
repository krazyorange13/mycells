#!/bin/bash

MODEL_NAME="thegrassisgreen_full-20000"
ANIM_IN="anim/${MODEL_NAME}-%03d.png"
ANIM_OUT="movies/${MODEL_NAME}.mp4"

ffmpeg -framerate 60 -i ${ANIM_IN} -vf "scale=8*iw:8*ih:flags=neighbor,format=yuv420p" -c:v libx264 -crf 17 -preset veryslow ${ANIM_OUT}
