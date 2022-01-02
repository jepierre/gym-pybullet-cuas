#!/bin/bash

# to rebuild with no cache
#docker build --no-cache -t uas_docker_env:1.0 -f docker/uas_docker_env.dockerfile .

docker build -t uas_docker_env:1.0 -f docker/uas_docker_env.dockerfile .
