#ifndef DATALOADER_H
#define DATALOADER_H

// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Core>
#include <string>
// --- CODE ---

class DataLoader
{

public:
    DataLoader(std::string filename, int batch_size);
};
#endif