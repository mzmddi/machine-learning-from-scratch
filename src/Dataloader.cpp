
// --- NOTES ---
/*
add window_size in the constructor of DataLoader
*/
// --- INCLUDE ---

#include <fstream>
#include <iostream>
#include <random>

#include "Dataloader.h"
#include "Batch.h"
// --- CODE ---

void DataLoader::normalize_batch(Batch &b)
{
    if (NORMALIZE)
    {
        b.batch_sample /= 255.0f;
    }
}

void DataLoader::print()
{
    std::cout << "\n===== DataLoader::print() =====" << std::endl;
}

void DataLoader::open_ifstream()
{

    std::ifstream file_1(this->filename);

    if (!file_1.is_open())
    {
        std::cout << "Could not open file " << filename << " in DataLoader::open_ifstream()" << std::endl;
        std::exit(1);
    };

    this->file = std::move(file_1);
}

void DataLoader::get_file_map()
{

    // ------------------------------------------------
    this->file.clear();

    std::string line;
    std::getline(this->file, line);
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> row;

    while (std::getline(ss, cell, ','))
    {
        row.push_back(cell);
    }
    this->num_of_features = row.size() - 1;

    // ---------------------------------------------------

    this->file.clear();
    this->file.seekg(0, std::ios::beg);

    uint64_t current_offset = this->file.tellg();

    std::string line_1;

    while (current_offset != static_cast<uint64_t>(-1) && std::getline(this->file, line_1))
    {
        this->file_map.push_back(current_offset);

        current_offset = this->file.tellg();
    }
}

void DataLoader::shuffle()
{
    std::shuffle(this->file_map.begin(), this->file_map.end(), this->rng);
}

void DataLoader::get_pool()
{

    this->pool_data.clear();
    this->pool_label.clear();

    int start = this->current_filemap_index_reached;
    int total_lines = this->file_map.size();

    if (start >= total_lines)
    {
        this->LAST_POOL = true;
        this->current_pool_size = 0;
        return;
    }

    int target_pool_size = this->batch_size * this->num_batch_in_pool;

    if (target_pool_size > total_lines - start)
    {
        target_pool_size = total_lines - start;
    }

    for (int i = 0; i < target_pool_size; i++)
    {

        this->file.clear();

        int current_line = start + i;

        if (current_line >= total_lines)
        {

            this->LAST_POOL = true;
            target_pool_size = i;
            break;
        };

        this->file.seekg(this->file_map[current_line]);
        std::string line;

        if (std::getline(this->file, line))
        {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            while (std::getline(ss, cell, ','))
            {
                row.push_back(cell);
            }

            for (int j = 0; j < row.size(); j++)
            {
                if (j == 0)
                {
                    this->pool_label.push_back(std::stof(row[j]));
                }

                else
                {
                    this->pool_data.push_back(std::stof(row[j]));
                };
            };
        };
    };

    this->current_pool_size = this->pool_data.size() / this->num_of_features;
    this->current_filemap_index_reached += target_pool_size;

    // --- TEST DEBUG ---
    // std::cout << "NEW POOL - current_pool_size:  " << this->current_pool_size << " - current_filemap_index_reached: " << this->current_filemap_index_reached << std::endl;
};

void DataLoader::initialize()
{
    this->open_ifstream();
    this->get_file_map();
    // this->get_pool();
    // get_pool should be called when the training starts
    // most likely when we first call get_batch()
}

DataLoader::DataLoader(std::string filename) : filename(filename)
{
    this->initialize();
}

DataLoader::DataLoader(std::string filename, int batch_size) : filename(filename)
{

    if (batch_size >= 2)
    {
        this->batch_size = batch_size;
    }
    else
    {
        std::cout << "BATCH_SIZE given to DATALOADER is less than 2." << std::endl;
        std::exit(0);
    }

    this->initialize();
}

void DataLoader::get_batch(Batch &b)
{
    if (this->current_pool_index_reached >= this->current_pool_size)
    {
        if (this->LAST_POOL)
        {
            b.all_batches_done = true;
            return;
        };

        this->get_pool();

        this->current_pool_index_reached = 0;
    }

    int target_amount = this->batch_size;

    if (target_amount > this->current_pool_size - this->current_pool_index_reached)
    {
        // if theres less samples in the pool than the batch size, just form a batch with the remaining
        // this happens only in the last pool, when whatever is remaining in the file is put in a pool

        target_amount = this->current_pool_size - this->current_pool_index_reached;
    }

    if (target_amount <= 0)
    {
        b.all_batches_done = true;
        return;
    }

    Eigen::MatrixXf b_data = Eigen::MatrixXf::Zero(this->num_of_features, target_amount);
    // the shape of the batch is (num_features, batch_size)
    // so each sample is a column in the matrix batch

    Eigen::MatrixXf b_label = Eigen::MatrixXf::Zero(10, target_amount);
    // column vector in the datatype of a matrix

    for (int i = 0; i < target_amount; i++)
    {

        for (int j = 0; j < this->num_of_features; j++)
        {
            // since pool_data is just floats, we need to count how many floats we want

            b_data(j, i) = this->pool_data[((this->current_pool_index_reached + i) * this->num_of_features) + j];
        };

        int label_position = static_cast<int>(this->pool_label[this->current_pool_index_reached + i]);

        b_label(label_position, i) = 1.0f;
    };

    this->current_pool_index_reached += target_amount;

    b.batch_sample = b_data;
    b.batch_label = b_label;

    this->normalize_batch(b);

    b.samples_in_batch = target_amount;
}

void DataLoader::test() {};

void DataLoader::reset(Batch &b)
{
    this->LAST_POOL = false;

    this->current_pool_size = 0;
    this->current_filemap_index_reached = 0;
    this->current_pool_index_reached = 0;

    b.all_batches_done = false;
}
