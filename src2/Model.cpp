// --- NOTES ---

// --- INCLUDES ---

#include <iostream>
#include <vector>
#include <string>

#include "Model.h"
#include "Card.h"
#include "Dataloader.h"
#include "Sample.h"

// --- CODE ---

void ml::Model::test() {};

void ml::Model::add(std::unique_ptr<ml::Card> c)
{
    this->cardstack.push_back(std::move(c));
};

void ml::Model::check_gates(std::vector<std::string> gates)
{
    bool one_wrong = false;

    for (auto g : gates)
    {
        auto it = this->gates.find(g);

        if (it == this->gates.end() || !(it->second))
        {

            std::cout << "[ERROR] GATE " << g << " IS NOT SET TO TRUE. CANNOT ACCESS NEXT PART OF THE CODE." << std::endl;

            if (!one_wrong)
            {
                one_wrong = true;
            };
        };
    };

    if (one_wrong)
    {
        std::exit(0);
    }
}

void ml::Model::check_all_gates()
{
    /*
    checks all the gates NEED TO DO THIS
    */

    std::vector<std::string> all_gate_names;

    all_gate_names.reserve(this->gates.size());

    for (const auto &[name, status] : this->gates)
    {
        all_gate_names.push_back(name);
    }

    this->check_gates(all_gate_names);
}

void ml::Model::print_architecture()
{
    std::cout << "==================" << std::endl;
    std::cout << "   ARCHITECTURE   " << std::endl;
    std::cout << "==================" << std::endl;
    int counter = 0;
    for (const auto &c : this->cardstack)
    {
        std::cout << "[" << counter << "] ";
        c->print();
        counter++;
    }
}

void ml::Model::set_test_dataloader(std::unique_ptr<DataLoader> dl)
{
    this->test_dl = std::move(dl);
};
void ml::Model::set_train_dataloader(std::unique_ptr<DataLoader> dl)
{
    this->train_dl = std::move(dl);
};

void ml::Model::do_forward_pass(Sample &s)
{

    for (auto &c : this->cardstack)
    {

        c->forward(s);
    }
};

void ml::Model::print_state()
{
    std::cout << "TOTAL EPOCHS: " << this->epoch << std::endl;
};

void ml::Model::do_backward_pass(Sample &s)
{

    for (int i = this->cardstack.size() - 1; i >= 0; i--)
    {
        this->cardstack[i]->backward(s);
    }
}

void ml::Model::do_an_update()
{
    for (auto &c : this->cardstack)
    {
        if (c->has_weight())
        {
            c->update(this->learning_rate);
        };
    };

    this->do_a_grad_reset();
};

void ml::Model::do_a_grad_reset()
{
    for (auto &c : this->cardstack)
    {
        if (c->has_weight())
        {
            c->reset_grad();
        };
    };
};

void ml::Model::do_one_epoch(Sample &s)
{

    /*
    one epoch includes the following:
        (1) - forward pass
        (2) - record and keep track of loss
        (3) - backwards pass
        (4) - update weights
    */

    int num_inputs = 0;

    double current_loss = 0.0f;

    int cached_epoch_reached = this->EPOCH_REACHED;

    while (true)
    {
        this->train_dl->next(s);

        if (s.done)
        {
            break;
        };

        s.features = s.features / s.max_value_in_features;

        num_inputs++;
        this->do_forward_pass(s);
        current_loss += s.loss;
        this->do_backward_pass(s);
        this->do_an_update();

        if (num_inputs % 100 == 0)
        {
            std::cout << "\rEPOCH [" << cached_epoch_reached << "] TRAINING: " << num_inputs << std::flush;
        };
    };

    this->EPOCH_TOTAL_CURRENT_LOSS = current_loss;
    this->EPOCH_AVG_CURRENT_LOSS = current_loss / num_inputs;
    std::cout << "  " << std::endl;
};

void ml::Model::do_one_test_pass()
{
    int num_inputs_test = 0;
    double current_loss_test = 0.0f;
    int correct_predictions = 0;

    Sample s_test;

    int cached_epoch_reached_test = this->EPOCH_REACHED;

    while (true)
    {
        this->test_dl->next(s_test);

        if (s_test.done)
        {
            break;
        }

        s_test.features = s_test.features / s_test.max_value_in_features;

        num_inputs_test++;
        this->do_forward_pass(s_test);
        current_loss_test += s_test.loss;

        if (num_inputs_test % 100 == 0)
        {
            std::cout << "\rEPOCH [" << cached_epoch_reached_test << "] TEST: " << num_inputs_test << std::flush;
        };

        int max_pred_idx = 0;
        int true_label_idx = 0;

        s_test.features.maxCoeff(&max_pred_idx);
        s_test.label.maxCoeff(&true_label_idx);

        if (max_pred_idx == true_label_idx)
        {
            correct_predictions++;
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / num_inputs_test * 100.0f;

    std::cout << " " << std::endl;
    std::cout << "EPOCH [" << this->EPOCH_REACHED << "] TEST AVG LOSS: " << current_loss_test / num_inputs_test << std::endl;
    std::cout << "EPOCH [" << this->EPOCH_REACHED << "] TEST ACCURACY: " << accuracy << "% (" << correct_predictions << "/" << num_inputs_test << ")" << std::endl;
    this->test_dl->reset(s_test);
}

void ml::Model::train()
{
    std::cout << "==================" << std::endl;
    std::cout << "  START TRAINING  " << std::endl;
    std::cout << "==================" << std::endl;

    // this->check_all_gates();

    while (true)
    {

        if (this->EPOCH_COUNTER == 0)
        {
            break;
        }

        Sample s;
        this->do_one_epoch(s);
        std::cout << "EPOCH [" << this->EPOCH_REACHED << "] AVG LOSS: " << this->EPOCH_AVG_CURRENT_LOSS << std::endl;
        this->train_dl->reset(s);

        if ((this->EPOCH_REACHED % this->test_interval) == 0)
        {
            this->do_one_test_pass();
        }
        this->EPOCH_REACHED++;
        this->EPOCH_COUNTER--;

        this->EPOCH_TOTAL_CURRENT_LOSS = 0.0f;
        this->EPOCH_AVG_CURRENT_LOSS = 0.0f;
    }

    std::cout << "DONE ALL EPOCHS" << std::endl;
};