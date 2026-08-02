#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Layers.h"
#include "Model.h"
#include "Dataloader.h"
#include "Sample.h"

int main()
{

    DataLoader dl = DataLoader("../assets/mnist_train.csv", "../assets/mnist_test.csv");

    // dl.print();

    Model m;

    m.set_dataloader(&dl);

    m.set_loss("CCE");

    m.add_linear(784, 500);
    m.add_ReLU();
    m.add_linear(500, 250);
    m.add_ReLU();
    m.add_linear(250, 10);
    m.add_Softmax();

    m.train();

    return 0;
}