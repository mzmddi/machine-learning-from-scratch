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

    DataLoader *dl_train;
    DataLoader *dl_test;

    bool initialised_sample = false;

    int epochs = 10;

    float lr = 0.0001f;

    int test_interval = 4;

public:
    Model();

    void add_linear(int in, int out);
    void add_ReLU();
    void add_Softmax();
    void print();
    void set_train(DataLoader *d) { this->dl_train = d; };
    void set_test(DataLoader *d) { this->dl_test = d; };
    void set_epochs(int s) { this->epochs = s; };

    void set_loss(std::string s);

    void set_lr(float s) { this->lr = s; };

    void train();

    void set_test_interval(int s) { this->test_interval = s - 1; };
    // i added - 1 because in the training loop, the start is 0, so the user will want to test every X epochs, but without the -1, it will test every X+1 because int i = 0

    void save(std::string filename);
    void load(std::string filename);

private:
    void forward_pass(Sample &s);
    // inner forward pass function.
    // a is temporary  and will be destroyed inside the train loop

    void back(Sample &s);
    // backprop function

    void update();

    void test();
};

#endif