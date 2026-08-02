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
};

#endif