FROM tensorflow/tensorflow:1.15.0-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y # && apt-get upgrade -y
RUN apt-get install -y \
            git \
            python3-matplotlib \
            python3-numpy \
            python3-sklearn \
            python3-opencv \
            vim less wget

#WORKDIR /yolo
#RUN wget https://pjreddie.com/media/files/yolov3.weights
#
#RUN git clone https://github.com/pjreddie/darknet
#WORKDIR /yolo/darknet
#RUN make

RUN pip3 install keras # opencv-contrib-python

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN mkdir -p /work
RUN chown -R user:user /work # /yolo

# Allow password-less 'root' login with 'su'
RUN passwd -d root
RUN sed -i 's/nullok_secure/nullok/' /etc/pam.d/common-auth

RUN echo $'#!/bin/bash\nPYTHONPATH=/tracker DEEPSORTHOME=/tracker YOLOHOME=/tracker python /tracker/count.py $*' > /usr/bin/count.sh

RUN chmod +x /usr/bin/count.sh

USER user

ENV PYTHONPATH=/tracker
WORKDIR /work

CMD /bin/bash

