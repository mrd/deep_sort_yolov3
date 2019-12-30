#!/bin/bash

# docker build -t deep_sort_yolov3 \
#   --build-arg USER_ID=$(id -u) \
#   --build-arg GROUP_ID=$(id -g) \
#   .
docker buildx build \
  --push \
  --platform linux/amd64,linux/arm/v7 \
  -t mrdanish/deep_sort_yolov3 \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .

