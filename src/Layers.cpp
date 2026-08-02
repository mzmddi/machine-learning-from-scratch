// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <cmath>

#include "Layers.h"

// --- CODE ---

void Layer::pass(Eigen::VectorXf &a)
{

    std::cout << "Failed to initiate layer::pass(). Fallback parent function." << std::endl;

    std::exit(1);
};

void Layer::backward(Eigen::VectorXf &delta)
{
    std::cout << "Failed to call backward() child. Fallback parent function." << std::endl;
    std::exit(1);
}

ReLU::ReLU()
{
    this->name = "ReLU";
};

void ReLU::pass(Eigen::VectorXf &a)
{

    this->input_cache.resize(a.size());
    this->input_cache = a;

    a = a.array().cwiseMax(0.0f);
};

void ReLU::backward(Eigen::VectorXf &delta)
{
    Eigen::VectorXf relu_grad = (this->input_cache.array() > 0.0f).cast<float>();
    delta = delta.cwiseProduct(relu_grad);
}

Linear::Linear(int in, int out) : in_feature(in), out_feature(out)
{
    this->W.setRandom(out, in + 1);
    // resized the mat W into in+1, out with rand vals [-1,1]
    // this is too big for ML so we need to scale it down using Xavier Glorot technique or way idk what its called

    float scale = std::sqrt(6.0f / (in + out));
    // setting up a very small value to scale the w and b with

    this->W *= scale;

    this->name = "Linear";

    this->grad = Eigen::MatrixXf::Zero(out, in + 1);
};

void Linear::print()
{

    // Arguments: precision, flags, coeff_sep, row_sep, row_prefix, row_suffix, mat_prefix, mat_suffix
    Eigen::IOFormat TwoDigitFmt(2, 0, " ", "\n", "", "", "", "");

    std::cout << "========================================\n"
              << " Linear Layer Summary\n"
              << "========================================\n"
              << " Weight Matrix Shape: (" << this->W.rows() << "x" << this->W.cols() << ")\n"
              << this->W.format(TwoDigitFmt) << "\n"
              << "========================================";
};

void Linear::pass(Eigen::VectorXf &a)
{

    this->input_cache.resize(a.size());
    this->input_cache = a;

    Eigen::VectorXf a_aug(a.size() + 1);
    a_aug << a, 1.0f;
    a = this->W * a_aug;
};

void Linear::backward(Eigen::VectorXf &delta)
{

    Eigen::VectorXf input_aug(this->input_cache.size() + 1);
    input_aug << this->input_cache, 1.0f;

    this->grad = delta * input_aug.transpose();

    Eigen::VectorXf delta_prev_aug = this->W.transpose() * delta;

    delta = delta_prev_aug.head(input_cache.size());
}

Softmax::Softmax()
{
    this->name = "Softmax";
};

void Softmax::pass(Eigen::VectorXf &a)
{
    a = (a.array() - a.maxCoeff()).exp();
    a /= a.sum();

    this->z.resize(a.size());
    this->z = a;
};

void Softmax::backward(Eigen::VectorXf &delta)
{
    int i = 1;

    // do nothing since this is the last layer and the equation is just collapsed to y^ - y calculated outside
    // we have to have something since the layers call this function for  all layers
}