#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Layers.h"
#include "Model.h"
#include "Dataloader.h"
#include "Sample.h"
#include <filesystem>
#include <vector>

int main()
{
    std::cout << " --- Start Dataloader ---" << std::endl;
    DataLoader dl = DataLoader("../assets/mnist_train.csv", "../assets/mnist_test.csv");
    std::cout << "Done creating the Dataloader" << std::endl;

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

    m.set_lr(0.0004f);

    m.set_epochs(2);

    std::cout << "--- start training ---" << std::endl;

    m.train();

    std::cout << "--- saving the model ---" << std::endl;

    m.save("test_save.bin");

    return 0;
}