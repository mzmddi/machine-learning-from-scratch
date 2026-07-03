#ifndef MODEL_H
#define MODEL_H

// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Core>
#include <vector>
#include "Layers.h"
#include "Dataloader.h"
// --- CODE ---

class Model
{
    int num_of_layers;
    std::vector<Layer *> layer_list;

public:
    Model();

    void add_linear(int in, int out);
    void add_ReLU();
    void add_Sigmoid();
    void print();

    void forward_pass(DataLoader &dataloader);

private:
    int check_layer_validity();
};

#endif