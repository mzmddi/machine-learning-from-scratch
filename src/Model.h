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

    DataLoader *dl;

public:
    Model();

    void add_linear(int in, int out);
    void add_ReLU();
    void add_Sigmoid();
    void print();
    void set_dataloader(DataLoader *d) { this->dl = d; };

private:
    void forward_pass(Eigen::VectorXf *a);
    // inner forward pass function.
    // a is temporary  and will be destroyed inside the train loop
};

#endif