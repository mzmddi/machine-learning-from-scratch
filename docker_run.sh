# this needs to be done after docker_build.sh has been run
# this should be done once your terminal is from the inside of the container


# --- setting the build environment ---

if [ -d "./build" ]; then
    cd build
    rm -rf *
else
    mkdir build
    cd build
fi

# --- Create the executable ---
cmake ../ && make
if [ $? -ne 0 ]; then
    echo "Building the executable failed!"
fi

clear 

./ml

echo " " 