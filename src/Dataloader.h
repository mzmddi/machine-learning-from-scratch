#ifndef DATALOADER_H
#define DATALOADER_H

// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Core>
#include <string>
#include <vector>
#include "Sample.h"
// --- CODE ---

class DataLoader
{
public:
    std::string filename;

    std::vector<float> data;
    std::vector<float> label;
    // data holders

    int num_of_features;
    bool feature_num_set = false;
    int num_of_sample = 0;

    int idx = 0;

    DataLoader(std::string filename);

    void print();

    void next(Sample &s);

    void reset(Sample &s);
};
#endif