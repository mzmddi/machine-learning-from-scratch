
// --- NOTES ---

// ---INCLUDES ---

#include "Model.h"
#include "Layers.h"
#include "Sample.h"
#include "Loss.h"
#include <iostream>
#include <regex>
#include <fstream>

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

void Model::add_Softmax()
{
    Softmax *s = new Softmax();
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
        std::cout << "layer [" << i << "]: ";
        this->layer_list[i]->print();
        std::cout << " " << std::endl;
    };
};

void Model::forward_pass()
{

    for (int i = 0; i < this->layer_list.size(); i++)
    {

        this->layer_list.at(i)->pass(this->s.features);
    }
};

void Model::set_loss(std::string s)
{
    if (s == "CCE")
    {

        CCE *cce = new CCE();

        this->loss = cce;
    }
    else
    {
        std::cout << "String did not match any Loss functions. Please revise the list." << std::endl;
        std::exit(1);
    }

    // for other losses, just append more `else if` here to add them.
    // must follow the pattern inherited by the parent class Loss.
}

void Model::back()
{

    Eigen::VectorXf ev = this->s.features - this->s.label;

    for (int i = static_cast<int>(this->layer_list.size()) - 1; i >= 0; --i)
    {
        // going through the layers backwards starting at the end.

        this->layer_list[i]->backward(ev);
    }
}

void Model::update()
{
    int size = this->layer_list.size();

    for (int i = 0; i < size; i++)
    {
        this->layer_list[i]->update(this->lr);
    }

    // std::cout << "done with updating the weights" << std::endl;
}

void Model::train()
{
    // void print_start_training();

    for (int i = 0; i < this->epochs; i++)
    {

        float sum_loss = 0.0f;

        int num_inputs = 0;

        while (true)
        {
            this->dl->next(this->s);
            if (this->s.done)
            {
                break;
            }

            num_inputs++;
            this->forward_pass();

            sum_loss += this->loss->compute_loss(this->s);

            // std::cout << "Loss: " << loss << std::endl;

            this->back();

            this->update();

            if (num_inputs % 50 == 0)
            {
                std::cout << "\rEPOCH [" << i + 1 << "] @ " << num_inputs << std::flush;
            }
        }
        this->dl->reset(this->s);
        std::cout << "\nEPOCH [" << i + 1 << "] \t AVG LOSS: " << sum_loss / num_inputs << std::endl;
    }
}

void Model::save(std::string filename)
{
    // 3 options : binary, json or csv
    // specified with the filename given

    std::regex pattern(R"(^(.+)(\.)([^.]+)$)");

    std::smatch matches;

    if (std::regex_match(filename, matches, pattern))
    {

        // std::ofstream out(matches[0].str(), std::ios::binary);

        std::ofstream out;
        out.open(filename, std::ios::binary);
        if (!out.is_open())
        {
            std::cout << "Cannot open binary file in model::save()" << std::endl;
            std::exit(1);
        }

        for (int i = 0; i < this->layer_list.size(); i++)
        {
            this->layer_list[i]->save(out);
        }

        out.flush();
        out.close();
        std::cout << "Model::save() DONE" << std::endl;
    }
    else
    {
        std::cout << "Filename given for model::save() does not match pattern." << std::endl;
        std::exit(1);
    }
}

void Model::load(std::string filename)
{
    // 0 => linear
    // 1 => ReLU
    // 2 => softmax

    std::regex pattern(R"(^(.+)(\.)([^.]+)$)");
    std::smatch matches;

    if (!std::regex_match(filename, matches, pattern) || matches[3] != "bin")
    {
        std::cerr << "Invalid filename format for Model::load(). Expected .bin extension." << std::endl;
        std::exit(1);
    }

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Could not open " << filename << " in model::load()" << std::endl;
        std::exit(1);
    }

    // logic: read layer_id -> create layer and add layer to model -> if linear, load the weights

    int layer_id = 0;
    while (in.read(reinterpret_cast<char *>(&layer_id), sizeof(int)))
    {
        Layer *new_layer = nullptr;

        switch (layer_id)
        {
        case 0:
        {
            int rows = 0;
            int cols = 0;

            in.read(reinterpret_cast<char *>(&rows), sizeof(int));
            in.read(reinterpret_cast<char *>(&cols), sizeof(int));

            int out_features = rows;
            int in_features = cols - 1;

            std::cout << "[DEBUG LOAD] Linear expected dims: " << in_features << "x" << out_features << std::endl;

            this->add_linear(in_features, out_features);

            std::cout << "[DEBUG LOAD] Stream pos BEFORE Linear::load(): " << in.tellg() << std::endl;
            this->layer_list.back()->load(in);
            std::cout << "[DEBUG LOAD] Stream pos AFTER Linear::load(): " << in.tellg() << std::endl;
            std::cout << "[DEBUG] - Linear layer done" << std::endl;
            break;
        }

        case 1:
        {
            this->add_ReLU();
            this->layer_list.back()->load(in);
            std::cout << "[DEBUG] - ReLU layer done" << std::endl;
            break;
        }

        case 2:
        {
            this->add_Softmax();
            this->layer_list.back()->load(in);
            std::cout << "[DEBUG] - Softmax layer done" << std::endl;
            break;
        }
        default:
            std::cout << "Could not read binary file in model::load()" << std::endl;
            std::exit(1);

            // add more cases in the future if new types of layers are created
        }
    }
    in.close();
    std::cout << "Model Successfully loaded from " << filename << std::endl;
}