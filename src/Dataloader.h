#ifndef DATALOADER_H
#define DATALOADER_H

// --- NOTES ---
/*
sample: 784 elements of train_data, 1 element of train_label
*/

// --- INCLUDES ---
#include <Eigen/Core>
#include <string>
#include <vector>
#include "Sample.h"
// --- CODE ---

class DataLoader
{
public:
    std::string train_filename;
    std::string test_filename;
    int batch_size;
    // param data

    std::vector<float> train_data;
    std::vector<float> train_label;
    std::vector<float> test_data;
    std::vector<float> test_label;
    // data holders

    int num_of_features;
    bool feature_num_set = false;
    int num_of_train_sample = 0;
    int num_of_test_sample = 0;
    int train_data_index = 0;

    int train_idx = 0;

    DataLoader(std::string train_filename, std::string test_filename, int batch_size);

    void print();

    Sample get_train_sample();
};
#endif