# this file is for when the program runs natively on apple m2
# running on m2, the program can use apple's special hardware
# the other version is on docker (refering to the files docker_build.sh and docker_run.sh)

mkdir build
cd build

cmake ../ 

make -j$(sysctl -n hw.ncpu)

if [ $? -ne 0 ]; then
    echo "Building the executable failed!"
    exit 1
fi

if [ -f "./ml" ]; then
clear
./ml
fi

echo " " 