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

ReLU::ReLU()
{
    this->name = "ReLU";
};

void ReLU::pass(Eigen::VectorXf &a)
{
    a = a.cwiseMax(0.0f);

    this->z.resize(a.size());
    this->z = a;
};

Linear::Linear(int in, int out) : in_feature(in), out_feature(out)
{
    this->W.setRandom(in + 1, out);
    // resized the mat W into in+1, out with rand vals [-1,1]
    // this is too big for ML so we need to scale it down using Xavier Glorot technique or way idk what its called

    float scale = std::sqrt(6.0f / (in + out));
    // setting up a very small value to scale the w and b with

    this->W *= scale;

    this->name = "Linear";

    this->grad = Eigen::MatrixXf::Zero(in + 1, out);
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

    Eigen::VectorXf a_aug(a.size() + 1);
    a_aug << a, 1.0f;
    a = this->W.transpose() * a_aug;

    this->act.resize(a.size());
    this->act = a;
};

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
