// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <cmath>
#include <fstream>

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
};

void Layer::update(float lr)
{
    std::cout << "Failed to call update() child. Fallback to parent function" << std::endl;
    std::exit(1);
};

void Layer::save(std::ofstream &out)
{
    std::cout << "Failed to call save() child. Fallback to parent function." << std::endl;
    std::exit(1);
}

void Layer::load(std::ifstream &in)
{

    std::cout << "Failed to call load() child. Fallback to parent function." << std::endl;
    std::exit(1);
}
void Layer::print()
{

    std::cout << "Failed to call print() child. Fallback to parent function." << std::endl;
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
};

void ReLU::update(float lr) {};

void ReLU::save(std::ofstream &out)
{
    int layer_id = 1;
    out.write(reinterpret_cast<const char *>(&layer_id), sizeof(int));
}

void ReLU::load(std::ifstream &in) {}

void ReLU::print() { std::cout << this->name << std::endl; }

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

    std::cout << this->name << this->W.rows() << "x" << this->W.cols() << std::endl;
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
};

void Linear::update(float lr)
{
    this->W -= this->grad * lr;
    this->grad.setZero();
}

void Linear::save(std::ofstream &out)
{
    int layer_id = 0;
    out.write(reinterpret_cast<const char *>(&layer_id), sizeof(int));

    int rows = static_cast<int>(this->W.rows());
    int cols = static_cast<int>(this->W.cols());

    std::cout << "[DEBUG SAVE] Linear layer dimensions: " << rows << "x" << cols << std::endl;
    out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(int));

    std::streamsize num_floats = static_cast<std::streamsize>(rows) * cols;
    out.write(reinterpret_cast<const char *>(this->W.data()), num_floats * sizeof(float));
};

void Linear::load(std::ifstream &in)
{

    std::streamsize num_floats = static_cast<std::streamsize>(this->W.rows()) * this->W.cols();
    std::cout << "[DEBUG LINEAR LOAD] Matrix W rows=" << this->W.rows()
              << " cols=" << this->W.cols()
              << " reading " << num_floats * sizeof(float) << " bytes" << std::endl;

    in.read(reinterpret_cast<char *>(this->W.data()), num_floats * sizeof(float));
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

void Softmax::backward(Eigen::VectorXf &delta) {};

void Softmax::update(float lr) {};

void Softmax::save(std::ofstream &out)
{
    int layer_id = 2;
    out.write(reinterpret_cast<const char *>(&layer_id), sizeof(int));
};
void Softmax::load(std::ifstream &in) {};

void Softmax::print() { std::cout << this->name << std::endl; }