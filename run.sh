#!/bin/bash

# docker run --rm -it --entrypoint /bin/bash uas_docker_env:1.0
xhost local:dev
docker run -it --net=host --privileged --rm -e "DISPLAY=$DISPLAY" -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v "$(pwd)"/gym-pybullet-drones:/home/dev/workspace/gym-pybullet-drones  uas_docker_env:1.0