
// --- NOTES ---

// ---INCLUDES ---

#include "Model.h"
#include "Layers.h"
#include <iostream>

// --- CODE ---

Model::Model() {};

void Model::add_linear(int in, int out)
{

    // add some verif to check if previous out == incoming in

    Linear *l = new Linear(in, out);

    this->layer_list.push_back(l);

    // std::cout << "DEBUG: Inside add_linear, size is now " << this->layer_list.size() << std::endl;
}

void Model::add_ReLU()
{
    ReLU *r = new ReLU();
    this->layer_list.push_back(r);
}

void Model::add_Sigmoid()
{
    Sigmoid *s = new Sigmoid();
    this->layer_list.push_back(s);
}

void Model::print()
{
    if (this->layer_list.empty())
    {
        std::cout << "Number of layers: 0" << std::endl;
        return;
    };

    std::cout << "Number of layers: " << this->layer_list.size() << std::endl;

    for (int i = 0; i < this->layer_list.size(); i++)
    {
        if (this->layer_list.at(i)->get_name() == "Linear")
        {
            std::cout << "layer [" << i << "]: " << this->layer_list.at(i)->get_name() << " in=" << this->layer_list.at(i)->in_feature << " out=" << this->layer_list.at(i)->out_feature << std::endl;
        }
        else if (this->layer_list.at(i)->get_name() == "ReLU")
        {
            std::cout << "layer [" << i << "]: " << this->layer_list.at(i)->get_name() << std::endl;
        }
        else if (this->layer_list.at(i)->get_name() == "Sigmoid")
        {
            std::cout << "layer [" << i << "]: " << this->layer_list.at(i)->get_name() << std::endl;
        };
    };
};

void Model::forward_pass(DataLoader &dataloader) {};