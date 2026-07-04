#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Layers.h"
#include "Model.h"
#include "Dataloader.h"
#include "Sample.h"

int main()
{

    // Eigen::Matrix<int, 3, 3> A;
    // A << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    // Eigen::Matrix<int, 3, 3> B;
    // B << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    // Eigen::Matrix<int, 3, 3> C = A.transpose() * B;

    // std::cout << C << std::endl;

    // Eigen::VectorXf r = Eigen::VectorXf::Random(2);

    // Linear l = Linear(5, 8);
    // // Linear m = Linear(8, 10);
    // // Linear n = Linear(10, 2);
    // // Linear o = Linear(2, 1);

    // Sigmoid sig = Sigmoid();

    // sig.pass(&r);

    // std::cout << r;
    // ReLU relu = ReLU();

    // std::cout << "\n\n"
    //           << r.transpose() << "\n\n";

    // l.pass(&r);

    // std::cout << r.transpose() << "\n\n";

    // relu.pass(&r);

    // std::cout << r.transpose() << "\n\n";

    // m.pass(&r);

    // std::cout << r.transpose() << "\n\n";

    // relu.pass(&r);

    // std::cout << r.transpose() << "\n\n";

    // n.pass(&r);

    // std::cout << r.transpose() << "\n\n";

    // relu.pass(&r);

    // std::cout << r.transpose() << "\n\n";

    // o.pass(&r);

    // std::cout << r.transpose();

    // Model m = Model();

    // m.add_linear(64, 512);
    // m.add_ReLU();
    // m.add_linear(512, 256);
    // m.add_ReLU();
    // m.add_linear(256, 10);
    // m.add_Sigmoid();

    // m.print();

    DataLoader dl = DataLoader("../assets/mnist_train.csv", "../assets/mnist_test.csv", 10);

    // Sample s = dl.get_sample();

    // std::cout << s.features.size() << std::endl;

    // std::cout << s.label << std::endl;

    Model m = Model();

    m.set_dataloader(&dl);

    return 0;
}