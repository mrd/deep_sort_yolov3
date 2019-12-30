#!/bin/bash

docker build -t guess_translation \
  -f Dockerfile.guess_translation \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .
# docker buildx build \
#   -f Dockerfile.guess_translation \
#   --push \
#   --platform linux/amd64,linux/arm/v7 \
#   -t mrdanish/guess_translation \
#   --build-arg USER_ID=$(id -u) \
#   --build-arg GROUP_ID=$(id -g) \
#   .
