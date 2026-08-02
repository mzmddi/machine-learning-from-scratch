#ifndef MODEL_H
#define MODEL_H

// --- NOTES ---

// --- INCLUDES ---
#include <Eigen/Core>
#include <vector>
#include "Layers.h"
#include "Dataloader.h"
#include "Sample.h"
#include "Loss.h"
// --- CODE ---

class Model
{
    int num_of_layers;
    std::vector<Layer *> layer_list;

    Loss *loss;

    DataLoader *dl;

    Sample s;

    bool initialised_sample = false;

    int epochs = 2;

public:
    Model();

    void add_linear(int in, int out);
    void add_ReLU();
    void add_Softmax();
    void print();
    void set_dataloader(DataLoader *d) { this->dl = d; };
    void set_epochs(int s) { this->epochs = s; };

    void set_loss(std::string s);

    void train();

private:
    void forward_pass();
    // inner forward pass function.
    // a is temporary  and will be destroyed inside the train loop
};

#endif