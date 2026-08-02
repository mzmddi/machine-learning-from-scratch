
#include "Loss.h"
#include "Sample.h"
#include <Eigen/Core>
#include <iostream>

float Loss::compute_loss(Sample &s)
{

    std::cout << "Failed to call loss child." << std::endl;

    std::exit(1);
};

float CCE::compute_loss(Sample &s)
{

    const float eps = 1e-7f;
    // threshold to become 0

    Eigen::VectorXf y_pred_clamped = s.features.array().cwiseMax(eps).cwiseMin(1.0f - eps);

    float loss = -(s.label.array() * y_pred_clamped.array().log()).sum();

    return loss;
};