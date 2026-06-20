// --- NOTES ---

// --- INCLUDES ---
#include "Matrix.h"

// --- CODE ---

mymath::Matrix::Matrix(int target_r, int target_c) : r(target_r), c(target_c), v(target_r * target_c)
{
    this->name = "unvalued";
};

mymath::Matrix::Matrix(int target_r, int target_c, const std::vector<float> &values) : r(target_r), c(target_c), v(target_r * target_c)
{

    int values_size = values.size();
    // using this variable a lot lets have it instead of continuously getting it from the std::vector

    if (values_size > r * c)
    {
        std::cout << "Matrix shape is less than number of values given. Matrix creation not possible due to overflow." << std::endl;
        std::exit(1);
        // if too many values are provided for the given size, error
    }

    for (int i = 0; i < values_size; i++)
    {
        this->set_value(i, values[i]);
    }

    this->name = "valued";
}

mymath::Matrix mymath::Matrix::Zero(int target_r, int target_c)
{

    Matrix temp(target_r, target_c);
    temp.set_name("zero");

    for (int i = 0; i < target_r * target_c; i++)
    {
        temp.set_value(i, 0.0f);
    }
    return temp;
};

mymath::Matrix mymath::Matrix::Identity(int target_p)
{
    Matrix temp = Matrix::Zero(target_p, target_p);

    temp.set_name("identity");

    for (int i = 0; i < target_p; i++)
    {
        temp.set_value(i * target_p + i, 1.0f);
    }
    return temp;
}

void mymath::Matrix::print()

{
    std::cout << std::endl;

    int size = this->get_shape()[0] * this->get_shape()[1];
    int row = this->get_shape()[0];
    int column = this->get_shape()[1];

    for (int i = 0; i < size; i++)
    {

        if (i % column == 0)
        {
            std::cout << std::endl;
        }
        std::cout << this->get_value(i) << " ";
    }
}

mymath::Matrix mymath::operator+(const mymath::Matrix &one, const mymath::Matrix &two)
{
    if (one.get_shape() != two.get_shape())
    {
        std::cout << "\nMatrices are not the same shape. Cannot perform addition." << std::endl;
        std::exit(1);
    }

    if (one.get_name() == "unvalued" || two.get_name() == "unvalued")
    {
        std::cout << "\nOne or both matrices have not been assigned values yet. Cannot perform addition." << std::endl;
        std::exit(1);
    }

    mymath::Matrix temp = mymath::Matrix::Zero(one.get_shape()[0], one.get_shape()[1]);

    int size = temp.get_shape()[0] * temp.get_shape()[1];

    for (int i = 0; i < size; i++)
    {
        temp.set_value(i, one.get_value(i) + two.get_value(i));
    };

    return temp;
}

template <typename T>
mymath::Matrix mymath::operator*(const mymath::Matrix &one, T scalar)
{
    mymath::Matrix temp = mymath::Matrix::Zero(one.get_shape()[0], one.get_shape()[1]);

    int size = one.get_shape()[0] * one.get_shape()[1];

    for (int i = 0; i < size; i++)
    {
        temp.set_value(i, one.get_value(i) * scalar);
    }

    return temp;
}

mymath::Matrix mymath::operator-(const mymath::Matrix &one, const mymath::Matrix &two)
{
    if (one.get_shape() != two.get_shape())
    {
        std::cout << "\nMatrices are not the same shape. Cannot perform substraction." << std::endl;
        std::exit(1);
    }

    if (one.get_name() == "unvalued" || two.get_name() == "unvalued")
    {
        std::cout << "\nOne or both matrices have not been assigned values yet. Cannot perform substraction" << std::endl;
        std::exit(1);
    }

    mymath::Matrix temp = mymath::Matrix::Zero(one.get_shape()[0], one.get_shape()[1]);

    int size = temp.get_shape()[0] * temp.get_shape()[1];

    for (int i = 0; i < size; i++)
    {
        temp.set_value(i, one.get_value(i) - two.get_value(i));
    };

    return temp;
}

mymath::Matrix mymath::operator%(const mymath::Matrix &one, const mymath::Matrix &two)
{
    if (one.get_shape()[1] != two.get_shape()[0])
    {
        std::cout << "\nThe two matrices are not compatible in shape. Cannot perform MM multiplication." << std::endl;
        std::exit(1);
    }

    if (one.get_name() == "unvalued" || two.get_name() == "unvalued")
    {
        std::cout << "\nOne or both matrices have not been assigned values yet. Cannot perform MM multiplication." << std::endl;
        std::exit(1);
    }

    int r1 = one.get_shape()[0];
    int c1 = one.get_shape()[1];
    int r2 = two.get_shape()[0];
    int c2 = two.get_shape()[1];

    mymath::Matrix temp = mymath::Matrix::Zero(r1, c2);

    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {

            int current_sum = 0.0f;

            for (int p = 0; p < c1; p++)
            {
                current_sum += one.get_value(i * c1 + p) * two.get_value(p * c2 + j);
            }

            temp.set_value(i * c2 + j, current_sum);
        }
    }

    return temp;
}

mymath::Matrix mymath::Matrix::t()
{

    int size = this->get_shape()[0] * this->get_shape()[1];

    int r = this->get_shape()[0];
    int c = this->get_shape()[1];

    mymath::Matrix temp = mymath::Matrix::Zero(c, r);

    for (int i = 0; i < c; i++)
    {

        std::vector<float> column(r);

        for (int j = 0; j < r; j++)
        {
            column[j] = this->get_value(j * c + i);
        }

        for (int p = 0; p < r; p++)
        {

            temp.set_value(i * r + p, column[p]);
        }
    }

    return temp;
}
// --- TEMPLATE BUILD SPECIFICATIONS ---

template mymath::Matrix mymath::operator*(const mymath::Matrix &, int);
template mymath::Matrix mymath::operator*(const mymath::Matrix &, float);
template mymath::Matrix mymath::operator*(const mymath::Matrix &, double);
