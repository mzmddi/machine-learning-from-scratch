# The base operating system to use
FROM ubuntu:20.04

RUN rm -rf /var/lib/apt/lists/*
# removing useless stuff that comes with the docker container

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake curl

WORKDIR /workspace
# setting the wrkdir to workspace so it's consistent

CMD ["/bin/bash"]
# opening a terminal
