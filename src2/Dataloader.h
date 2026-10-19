#ifndef DATALOADER_H
#define DATALOADER_H

// --- NOTES ---
/*

size of the inputs need to be consistent across all inputs.

*/
// --- INCLUDES ---
#include <Eigen/Core>
#include <string>
#include <vector>
#include "Sample.h"
#include <fstream>
// --- CODE ---

class DataLoader
{
public:
    bool LAST_WINDOW = false;


    std::string filename;
    // filename given at the constructor

    std::ifstream file;
    // the ifstream file keeping the connection with the file open

    std::vector<float> data;
    std::vector<float> label;
    // data holders

    int feature_size = 0;
    // the size of the sample, the number of features per sample

    int num_of_features; // to be removed, but need to change the entire codebase

    bool feature_num_set = false;

    int num_of_sample = 0;

    std::vector<float> window_data;
    std::vector<float> window_label;
    int window_size;
    // window_size = 100 is the default

    int file_line_idx = 0;
    // the index corresponding to the location of the file where the current window starts

    int curr_sample_idx = 0;
    // the current sample index reletive to the window

    std::vector<uint64_t> offset_map;

    DataLoader(std::string filename, int window_size = 200);

    void print();

    void next(Sample &s);

    void reset(Sample &s);

    void set_window_size(int s) { this->window_size = s; };
    // when called, this should recall get_window, so that window is updated with new size

    void get_feature_size();
    // reads the first line of the

    void get_offset_map();
    // map the entire dataset and record where the new lines start

    void get_window();

    void open_ifstream();

    ~DataLoader() { this->file.close(); };
};
#endif