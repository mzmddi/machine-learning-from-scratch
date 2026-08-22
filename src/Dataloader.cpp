
// --- NOTES ---

// --- INCLUDE ---

#include "Dataloader.h"
#include <fstream>
#include <iostream>
#include "Sample.h"
// --- CODE ---

DataLoader::DataLoader(std::string filename) : filename(filename)
{

    std::ifstream file(filename);
    // needs to be a csv for this project

    if (!file.is_open())
    {
        std::cout << "Could not open file " << filename << std::endl;
        std::exit(1);
    };

    std::string line;

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

        this->label.push_back(std::stof(row[0]));

        for (int i = 1; i < row.size(); i++)
        {
            // skip the row[0] since the first element is the label
            this->data.push_back(std::stof(row[i]));
        }

        this->num_of_sample += 1;
    }

    file.close();
}

void DataLoader::print()
{
    std::cout << "\n===== DataLoader::print() =====" << std::endl;
    std::cout << "this->filename: " << this->filename << std::endl;
    std::cout << "\n"
              << std::endl;
    std::cout << "this->data.size(): " << this->data.size() << std::endl;
    std::cout << "this->label.size(): " << this->label.size() << std::endl;
    std::cout << "\n"
              << std::endl;
    std::cout << "this->num_of_features: " << this->num_of_features << std::endl;
    std::cout << "this->num_of_sample: " << this->num_of_sample << std::endl;
}

void DataLoader::next(Sample &s)
{

    if (this->idx == this->num_of_sample || this->idx == 2000)
    {

        // THE OR SECTION OF THE IF STATEMENT IF JUST HERE FOR DEVELOPMENT BECAUSE I DO NOT WANT TO WAIT FOR ALL 60k INPUTS TO PASS!!!
        // SHOULD BE REMOVED ONCE EVERYTHING IS DONE
        s.done = true;
    }
    else
    {

        int label_position = this->label[this->idx];

        Eigen::VectorXf l(10);

        for (int i = 0; i < 10; i++)
        {
            if (i == label_position)
            {
                l[i] = 1.0f;
            }
            else
            {
                l[i] = 0.0f;
            }
        }

        s.label = l;

        const float *feature_ptr = this->data.data() + (this->idx * this->num_of_features);

        s.original = Eigen::Map<const Eigen::VectorXf>(feature_ptr, this->num_of_features);

        s.features = Eigen::Map<const Eigen::VectorXf>(feature_ptr, this->num_of_features);

        this->idx += 1;
    }
}

void DataLoader::reset(Sample &s)
{
    this->idx = 0;

    s.done = false;
}