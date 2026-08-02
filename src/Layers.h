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

    virtual void pass(Eigen::VectorXf &a);
};

class ReLU : public Layer
{
public:
    ReLU();

    Eigen::VectorXf z;

    void pass(Eigen::VectorXf &a) override;
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

public:
    Eigen::MatrixXf W;
    Eigen::MatrixXf grad;
    Eigen::VectorXf act;
    int in_feature;
    int out_feature;
    Linear(int in, int out);

    void pass(Eigen::VectorXf &a) override;

    void print();
};

class Softmax : public Layer
{
public:
    Softmax();
    Eigen::VectorXf z;
    void pass(Eigen::VectorXf &a) override;
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