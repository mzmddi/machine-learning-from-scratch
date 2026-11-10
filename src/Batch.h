#ifndef BATCH_H
#define BATCH_H

#include <Eigen/Core>

struct Batch
{
    int samples_in_batch = 0;

    float max_value_in_features = 255.0f;

    bool all_batches_done = false;

    Eigen::VectorXf batch_loss;

    Eigen::MatrixXf batch_sample;
    Eigen::MatrixXf batch_label;
    Eigen::MatrixXf batch_grad_prod;
};

#endif

/*

--- NOTES ---

batch_sample

shape = (M, N)
M = features per sample
N = num of samples
vibes => each column is a specific sample
ex => (20, 10) the batch has 10 sample, each with 20 features


*/