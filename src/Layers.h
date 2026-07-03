#ifndef LAYERS_H
#define LAYERS_H

// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Dense>
#include <string>
#include <iostream>

// --- CODE ---

class Layer
{
public:
    int in_feature;
    int out_feature;
    std::string name;

    std::string get_name() const { return this->name; }

    void pass(Eigen::VectorXf *a);
};

class ReLU : public Layer
{
public:
    ReLU();

    void pass(Eigen::VectorXf *a);
};

class Linear : public Layer
{

    /*
    a_(n) = W^(T) * a_(n-1) + b
    a_(n-1) is the output of the previous layer
    a_0 would therefore be the input

    W       : (D, M)
    a_(n-1) : (D, 1) column vector
    a_(n)   : (M, 1) output to the next layer
    b       : (M, 1) bias

    the shape of W is influenced by the shape of the previous output and the shape of the next output i want, so both by in and out

    D+1 is the in
    M is the out
    so => Shape of W = (in + 1, out)
    */

    Eigen::MatrixXf W;
    Eigen::VectorXf b;

public:
    int in_feature;
    int out_feature;
    Linear(int in, int out);

    void pass(Eigen::VectorXf *a);

    void print();
};

class Sigmoid : public Layer
{
public:
    Sigmoid();
    void pass(Eigen::VectorXf *a);
};

#endif

/*

Layer class - parent class
void pass()
* Linear class
* * initialize()
* ReLU()
* Sigmoid

*/