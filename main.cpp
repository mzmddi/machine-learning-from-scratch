#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "Dataloader.h"
#include "Model.h"
#include "Card.h"

int main(int argc, char *argv[])
{

#ifdef _WIN32
    std::system("cls"); // Windows
#else
    std::system("clear"); // Linux / macOS
#endif

    ml::Model m = ml::ModelBuilder()
                      .train_dataloader_filename("../assets/mnist_train.csv", 8)
                      .test_dataloader_filename("../assets/mnist_test.csv", 8)
                      .shuffle()
                      .epoch(30)
                      .learning_rate(1e-5f)
                      .test_interval(3)
                      .LINEAR(784, 128)
                      .RELU()
                      .LINEAR(128, 64)
                      .RELU()
                      .LINEAR(64, 10)
                      .SOFTMAX()
                      .CCE();

    m.shape_check();

    m.train();
}