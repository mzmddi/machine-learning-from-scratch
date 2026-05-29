# The base operating system to use
FROM ubuntu:20.04

RUN rm -rf /var/lib/apt/lists/*
# removing useless stuff that comes with the docker container

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake curl

RUN mkdir -p /usr/local/include/Eigen
# making the directory of the Eigen library

RUN curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz -C /usr/local/include/Eigen --strip-components=1
# downloading the header of the eigen library to keep it fast and small

WORKDIR /workspace
# setting the wrkdir to workspace so it's consistent

CMD ["/bin/bash"]
# opening a terminal
