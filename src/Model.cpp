
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

void Model::load(std::string filename) {}