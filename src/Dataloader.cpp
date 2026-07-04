
// --- NOTES ---

// --- INCLUDE ---

#include "Dataloader.h"
#include <fstream>
#include <iostream>
#include "Sample.h"
// --- CODE ---

DataLoader::DataLoader(std::string train_filename, std::string test_filename, int batch_size) : train_filename(train_filename), test_filename(test_filename), batch_size(batch_size)
{

    std::ifstream file(train_filename);
    // needs to be a csv for this project

    if (!file.is_open())
    {
        std::cout << "Could not open file " << train_filename << std::endl;
        std::exit(1);
    };

    std::string line;

    // there's no header line in the test file so we will not concider it but if there was, we need the fist line

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;

        std::vector<std::string> row;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }

        if (!this->feature_num_set)
        {
            this->num_of_features = row.size() - 1;
        }

        this->train_label.push_back(std::stof(row[0]));

        for (int i = 0; i < row.size(); i++)
        {
            this->train_data.push_back(std::stof(row[i]));
        }

        this->num_of_train_sample += 1;
    }

    file.close();

    // ======================= test data ==========

    std::ifstream file2(test_filename);
    // needs to be a csv for this project

    if (!file2.is_open())
    {
        std::cout << "Could not open file " << test_filename << std::endl;
        std::exit(1);
    };

    // there's no header line in the test file so we will not concider it but if there was, we need the fist line

    while (std::getline(file2, line))
    {
        std::stringstream ss(line);
        std::string cell;

        std::vector<std::string> row;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }

        if (!this->feature_num_set)
        {
            this->num_of_features = row.size() - 1;
        }

        this->test_label.push_back(std::stof(row[0]));

        for (int i = 0; i < row.size(); i++)
        {
            this->test_data.push_back(std::stof(row[i]));
        }

        this->num_of_test_sample += 1;
    }
}

void DataLoader::print()
{
    std::cout << "\n===== DataLoader::print() =====" << std::endl;
    std::cout << "this->train_filename: " << this->train_filename << std::endl;
    std::cout << "this->test_filename: " << this->test_filename << std::endl;
    std::cout << "this->batch_size: " << this->batch_size << std::endl;
    std::cout << "this->train_data.size(): " << this->train_data.size() << std::endl;
    std::cout << "this->train_label.size(): " << this->train_label.size() << std::endl;
    std::cout << "this->test_data.size(): " << this->test_data.size() << std::endl;
    std::cout << "this->test_label(): " << this->test_label.size() << std::endl;
    std::cout << "this->num_of_features: " << this->num_of_features << std::endl;
    std::cout << "this->num_of_train_sample: " << this->num_of_train_sample << std::endl;
    std::cout << "this->num_of_test_sample: " << this->num_of_test_sample << std::endl;
}

Sample DataLoader::get_train_sample()
{
    Sample s;

    int row_stride = this->num_of_features;

    int start_of_row = this->train_idx * row_stride;

    s.label = this->train_label[train_idx];

    s.features.assign(this->train_data.begin() + start_of_row, this->train_data.begin() + start_of_row + row_stride);

    this->train_idx += 1;

    return s;
}