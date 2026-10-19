#ifndef SAMPLE_H
#define SAMPLE_H

#include <vector>
#include <Eigen/Core>

struct Sample
{
    Eigen::VectorXf original;
    Eigen::VectorXf features;
    Eigen::VectorXf label;
    bool done = false;
    float loss;
    Eigen::VectorXf grad_prod;
    float max_value_in_features = 255.0f;
};

#endif