# run these to build the container of the program
# this needs to be done first

docker build -f Dockerfile -t ml-env .

docker run -it --rm -v "${PWD}:/workspace" ml-env