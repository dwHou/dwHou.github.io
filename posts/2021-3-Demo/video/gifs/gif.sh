#!/usr/bin/env bash 
ffmpeg -i zoom-sr.mp4 -vframes 20 -vf "fps=1" -t 20 %d.png
ffmpeg -framerate 10 -i %d.png -vf "fps=10,scale=512:-1,palettegen" -y palette.png
ffmpeg -framerate 5 -i %d.png -i palette.png -filter_complex "fps=5,scale=512:-1 [x]; [x][1:v] paletteuse" -r 5 -vcodec gif output.gif