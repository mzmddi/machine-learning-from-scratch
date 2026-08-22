#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Layers.h"
#include "Model.h"
#include "Dataloader.h"
#include "Sample.h"
#include <filesystem>
#include <vector>

#include <fstream>

int main()
{
    std::cout << "Start" << std::endl;

    DataLoader dl_train = DataLoader("../assets/mnist_train.csv");

    std::cout << "train done" << std::endl;

    DataLoader dl_test = DataLoader("../assets/mnist_test.csv");

    std::cout << "all done" << std::endl;

    Model m;

    m.set_train(&dl_train);
    m.set_test(&dl_test);

    m.set_loss("CCE");

    m.add_linear(784, 500);
    m.add_ReLU();
    m.add_linear(500, 250);
    m.add_ReLU();
    m.add_linear(250, 10);
    m.add_Softmax();

    m.set_lr(0.00002f);

    m.set_epochs(50);

    m.set_test_interval(2);

    std::cout << "--- start training ---" << std::endl;

    m.train();

    // std::cout << "--- saving the model ---" << std::endl;

    // m.save("test_save.bin");

    // ---------------------------------------------

    // Model m;

    // m.load("../test_save.bin");

    // m.print();

    // return 0;

    // -------------------------------

    // std::fstream out;

    // out.open("testing.bin", std::ios::out | std::ios::binary);

    // // the std::ios::out specifies which direction -> out for writing the file, in for reading the file

    // std::vector<float> data;

    // for (int i = 0; i < 20; i++)
    // {
    //     data.push_back(static_cast<float>(i));
    // }

    // if (out)
    // {

    //     out.write(reinterpret_cast<char *>(&data), data.size() * sizeof(float));
    //     out.close();
    // }
    // else
    // {
    //     std::cout << "could not open file for writing" << std::endl;
    // }

    // std::fstream in;
    // in.open("testing.bin", std::ios::in | std::ios::binary);

    // if (in)
    // {
    //     while (true)
    //     {
    //         float x = -1.0f;

    //         in.read(reinterpret_cast<char *>(&x), sizeof(float));

    //         std::cout << x << " ";

    //         if (x == -1.0f)
    //         {
    //             std::exit(0);
    //         }
    //     }

    //     in.close();
    // }
    // else
    // {
    //     std::cout << "Could not open file for reading" << std::endl;
    // }

    // std::cout << " " << std::endl;

    // return 0;
}