
// --- NOTES ---
/*
add window_size in the constructor of DataLoader
*/
// --- INCLUDE ---

#include "Dataloader.h"
#include <fstream>
#include <iostream>
#include "Sample.h"
// --- CODE ---

void DataLoader::open_ifstream()
{

    // opens the file to read it, and stores the stream
    // with the stream constantly open, less time in the future having to keep opening the file
    // especially since we are doing the window method

    std::ifstream file_1(this->filename);

    if (!file_1.is_open())
    {
        std::cout << "Could not open file " << filename << "in DataLoader::open_ifstream()" << std::endl;
        std::exit(1);
    };

    this->file = std::move(file_1);
}

void DataLoader::get_feature_size()
{

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

    this->feature_size = row.size() - 1;
    // assuming that the label is the first value of the row, so we need to do -1 to account for that.
}

void DataLoader::get_offset_map()
{
    // THIS DOES NOT ACCOUNT FOR THE HEADER, IF HEADER IS PRESENT IT WILL READ IT AS WELL

    this->file.clear();
    this->file.seekg(0, std::ios::beg);

    uint64_t current_offset = this->file.tellg();

    std::string line;

    while (current_offset != static_cast<uint64_t>(-1) && std::getline(this->file, line))
    {
        this->offset_map.push_back(current_offset);

        current_offset = this->file.tellg();
    }
}

void DataLoader::get_window()
{

    this->window_data.clear();
    this->window_label.clear();
    // we need to clear the windows of the data and label so that whatever the new window is, it is fresh and the previous window is disgarded.

    int start = this->file_line_idx;

    int total_lines = this->offset_map.size();

    file.clear();
    // since we are using seekg(), we need to clear the file pointer

    for (int i = 0; i < this->window_size; i++)
    {
        int current_line = start + i;

        if (current_line >= total_lines)
        {

            // this->window_size = i + 1;
            // // new size of window since the last window does not have the same window space

            this->LAST_WINDOW = true;

            break;
            // break the loop early since it's th end of the offset_map
        }

        this->file.seekg(this->offset_map[current_line]);

        std::string line;

        if (std::getline(this->file, line))
        {
            // PROCESS THE STRING LINE HERE
            // FROM STRING TO FLOATS
            // TRAKE THE FIRST ONE TO BE THE lABEL
            //
            // CHECK THE DataLoader::DataLoader() for the sscan part

            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            // the row from the file is basically label + features

            while (std::getline(ss, cell, ','))
            {
                row.push_back(cell);
            }

            for (int j = 0; j < row.size(); j++)
            {
                if (j == 0)
                {
                    this->window_label.push_back(std::stof(row[j]));
                }

                else
                {
                    this->window_data.push_back(std::stof(row[j]));
                }
            }

            // now std::vector<string> contains the row from the file
        }
    }

    if (this->window_label.size() != (this->window_data.size() / this->feature_size))
    {
        std::cout << "Number of labels and Number of samples in the window does not match." << std::endl;
        std::cout << "Number of labels: " << this->window_label.size() << std::endl;
        std::cout << "Total number of features: " << this->window_data.size() << std::endl;
        std::cout << "Number of samples: " << this->window_data.size() / this->feature_size << std::endl;
        std::exit(1);
    }
}

DataLoader::DataLoader(std::string filename, int window_size) : filename(filename), window_size(window_size)
{

    this->open_ifstream(); // done

    this->get_feature_size(); // done

    this->get_offset_map(); // done

    this->get_window(); // done

    // the constructor ends with DataLoader::get_window() which loads the first batch of samples/label in the window.
    // get_window() clears both vectors and reloads the floats based on the last window_size index
    // the incrementation of the window to the next batch from the file is in the DataLoader::next() since we need to know when the window is done and that is done only when next() goes through the entire std::vector

    //
    //
    //
    // the next part is DataLoader::next() by using the window and the implementation of incrementing the window from the file.
}

void DataLoader::print()
{
    std::cout << "\n===== DataLoader::print() =====" << std::endl;
    std::cout << "this->filename: " << this->filename << std::endl;
    std::cout << "this->data.size(): " << this->data.size() << std::endl;
    std::cout << "this->label.size(): " << this->label.size() << std::endl;
    std::cout << "this->feature_size:" << this->feature_size << std::endl;
    std::cout << "this->num_of_features: " << this->num_of_features << std::endl;
    std::cout << "this->num_of_sample: " << this->num_of_sample << std::endl;
    std::cout << "this->window_data.size(): " << this->window_data.size() << std::endl;
    std::cout << "this->window_data.size() / this->feature_size: " << this->window_data.size() / this->feature_size << std::endl;
    std::cout << "this->window_label.size(): " << this->window_label.size() << std::endl;
    std::cout << "this->window_size" << this->window_size << std::endl;
    std::cout << "this->of" << std::endl;
}

void DataLoader::next(Sample &s)
{

    if (this->curr_sample_idx >= this->window_data.size() / this->feature_size)
    {
        // increment the file_line_idx to the next window size

        if (this->LAST_WINDOW)
        {
            // here -> end of window + last window, so the next() should terminate

            s.done = true;
            return;
        }
        // here -> end of window, but not the last window

        this->file_line_idx += this->window_data.size() / this->feature_size;
        // move the window of data one window_size over

        this->curr_sample_idx = 0;
        // reset the counter inside the window

        this->get_window();
        // get the new window with the new values defined just above.
    }

    int label_pos = this->window_label[this->curr_sample_idx];

    Eigen::VectorXf l(10);

    for (int i = 0; i < 10; i++)
    {
        if (i == label_pos)
        {
            l[i] = 1.0f;
        }
        else
        {
            l[i] = 0.0f;
        }
    }

    s.label = l;

    const float *feature_ptr = this->window_data.data() + (this->curr_sample_idx * this->feature_size);

    s.original = Eigen::Map<const Eigen::VectorXf>(feature_ptr, this->feature_size);

    s.features = Eigen::Map<const Eigen::VectorXf>(feature_ptr, this->feature_size);

    // std::cout << "[DEBUG] - Label: " << label_pos << " First feature: " << s.features[0] << " current_sample_idx: " << this->curr_sample_idx << std::endl;

    this->curr_sample_idx += 1;

    // -------- OLD code is below this line -------------
    //
    //
    //
    // if (this->idx == this->num_of_sample || this->idx == 2000)
    // {

    //     // THE OR SECTION OF THE IF STATEMENT IF JUST HERE FOR DEVELOPMENT BECAUSE I DO NOT WANT TO WAIT FOR ALL 60k INPUTS TO PASS!!!
    //     // SHOULD BE REMOVED ONCE EVERYTHING IS DONE
    //     s.done = true;
    // }
    // else
    // {

    //     int label_position = this->label[this->idx];

    //     Eigen::VectorXf l(10);

    //     for (int i = 0; i < 10; i++)
    //     {
    //         if (i == label_position)
    //         {
    //             l[i] = 1.0f;
    //         }
    //         else
    //         {
    //             l[i] = 0.0f;
    //         }
    //     }

    //     s.label = l;

    //     const float *feature_ptr = this->data.data() + (this->idx * this->num_of_features);

    //     s.original = Eigen::Map<const Eigen::VectorXf>(feature_ptr, this->num_of_features);

    //     s.features = Eigen::Map<const Eigen::VectorXf>(feature_ptr, this->num_of_features);

    //     this->idx += 1;
    // }
}

void DataLoader::reset(Sample &s)
{
    this->curr_sample_idx = 0;
    // resetting the current sample index back to 0

    this->file_line_idx = 0;
    // reset the index in the file, so that we restart at the begining of the file

    this->LAST_WINDOW = false;
    // reset the last_window to false so that it does not go into the last window logic

    s.done = false;
    // same thing the lastwindow thing, but this time the logic is for the entire forward pass

    this->get_window();
    // before we potentially start a new epoch, we fetch a new fresh batch of data from the start directly since we reset all the values
}