#ifndef DATALOADER_H
#define DATALOADER_H

// --- NOTES ---
/*

The constructor has 3 oblig values filename, batch size, window size
if the user wishes to provide the default value, the user should input 0 for the ints.

*/
// --- INCLUDES ---
#include <Eigen/Core>
#include <string>
#include <vector>
#include <fstream>
#include <random>

#include "Batch.h"
// --- CODE ---

class DataLoader
{

    // --- GATES ---

    std::unordered_map<std::string, bool> gates = {
        {"GATE_FILE_STREAM_SET", false},
        {"GATE_OFFSET_MAP_DONE", false}};

    // --- GENERAL CLASS DATA ---

    int batch_size = 1;
    int num_of_features;
    int num_batch_in_pool = 5;
    // original 100

    std::string filename;
    std::ifstream file;

    std::vector<uint64_t> file_map;

    std::mt19937 rng;

    // --- INTERNAL LOGIC DATA ---

    // things in this category may need to be reset at any given loop
    // basically this is fields for a given state of the system

    int current_pool_size = 0;
    int current_filemap_index_reached = 0; // for da pool
    int current_pool_index_reached = 0;    // for da batch

    std::vector<float> pool_data;
    std::vector<float> pool_label;

    bool LAST_POOL = false;

public:
    // all the APIs

    DataLoader(std::string filename);
    DataLoader(std::string filename, int batch_size);
    ~DataLoader() { this->file.close(); };

    void get_batch(Batch &b);

    void print();

    void shuffle();

    void test();

private:
    void initialize();
    void open_ifstream();
    void get_file_map();
    void get_pool();

    void next();
    void reset();
    // std::vector<float> data;
    // std::vector<float> label;
    // // data holders

    // int feature_size = 0;
    // // the size of the sample, the number of features per sample

    // int num_of_features; // to be removed, but need to change the entire codebase

    // bool feature_num_set = false;

    // int num_of_sample = 0;

    // std::vector<float> window_data;
    // std::vector<float> window_label;

    // int file_line_idx = 0;
    // // the index corresponding to the location of the file where the current window starts

    // int curr_sample_idx = 0;
    // // the current sample index reletive to the window

    // std::vector<uint64_t> offset_map;

    // void print();

    // void set_window_size(int s) { this->window_size = s; };
    // // when called, this should recall get_window, so that window is updated with new size

    // void get_feature_size();
    // // reads the first line of the

    // void get_offset_map();
    // // map the entire dataset and record where the new lines start

    // void get_window();

    // void open_ifstream();
};
#endif