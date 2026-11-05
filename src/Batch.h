#ifndef BATCH_H
#define BATCH_H

#include <vector>
#include <Eigen/Core>

struct Batch
{

    float max_value_in_features = 255.0f;

    bool all_batches_done = false;

    Eigen::MatrixXf batch_sample;
    Eigen::MatrixXf batch_label;
    Eigen::MatrixXf batch_loss;
    Eigen::MatrixXf batch_grad_prod;
};

#endif