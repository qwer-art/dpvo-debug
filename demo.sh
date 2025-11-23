#!/bin/bash

if [[ $1=="debug" ]]; then
    export PYDEBUG=" -m debugpy --wait-for-client --listen localhost:3197 "
    shift
else
    export PYDEBUG=""
fi

python_file=/home/jerett/Project/DPVO/demo.py
project_path=/home/jerett/Project/DPVO
python $PYDEBUG "$python_file" --imagedir=$project_path/movies/IMG_0492.MOV --calib=$project_path/calib/iphone.txt --stride=5 --viz