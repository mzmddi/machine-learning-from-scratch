#include <iostream>
#include "Matrix.h"

int main()
{

    std::vector<float> test{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> test2{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::vector<float> test3{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    mymath::Matrix m = mymath::Matrix(2, 3, test3);
    mymath::Matrix n = mymath::Matrix(2, 3, test3);

    mymath::Matrix g = mymath::Matrix::Identity(2);

    m.print();

    mymath::Matrix d = n.t();

    d.print();

    mymath::Matrix c = (m % n.t()) % (g * 2);

    c.print();
}