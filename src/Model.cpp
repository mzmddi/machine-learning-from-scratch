// --- NOTES ---

// --- INCLUDES ---

#include <iostream>
#include <vector>
#include <string>

#include "Model.h"
#include "Card.h"
#include "Dataloader.h"
#include "Batch.h"

// --- CODE ---

// --- CAR BUILDER CODE ---

ml::ModelBuilder &ml::ModelBuilder::train_dataloader_filename(std::string filename)
{

    if (model.gates["GATE_TRAIN_DL"])
    {
        // need to log this

        std::cout << "TRAIN_DATALOADER is already set for the model. Skipping double assignment." << std::endl;
        return *this;
    }

    model.train_dl = std::move(std::make_unique<DataLoader>(filename));

    model.gates["GATE_TRAIN_DL"] = true;

    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::test_dataloader_filename(std::string filename)
{
    if (model.gates["GATE_TEST_DL"])
    {
        // need to log this

        std::cout << "TEST_DATALOADER is already set for the model. Skipping double assignment." << std::endl;
        return *this;
    }
    model.test_dl = std::move(std::make_unique<DataLoader>(filename));

    model.gates["GATE_TEST_DL"] = true;

    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::train_dataloader_filename(std::string filename, int batch_size)
{

    if (model.gates["GATE_TRAIN_DL"])
    {
        // need to log this

        std::cout << "TRAIN_DATALOADER is already set for the model. Skipping double assignment." << std::endl;
        return *this;
    }

    model.train_dl = std::move(std::make_unique<DataLoader>(filename, batch_size));

    model.gates["GATE_TRAIN_DL"] = true;

    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::test_dataloader_filename(std::string filename, int batch_size)
{
    if (model.gates["GATE_TEST_DL"])
    {
        // need to log this

        std::cout << "TEST_DATALOADER is already set for the model. Skipping double assignment." << std::endl;
        return *this;
    }
    model.test_dl = std::move(std::make_unique<DataLoader>(filename, batch_size));

    model.gates["GATE_TEST_DL"] = true;

    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::learning_rate(float lr)
{
    model.learning_rate = lr;
    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::test_interval(int ti)
{
    model.test_interval = ti;
    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::epoch(int e)
{
    model.epoch = e;
    model.EPOCH_COUNTER = e;

    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::shuffle()
{
    if (!model.gates["GATE_TEST_DL"] && !model.gates["GATE_TRAIN_DL"])
    {
        // log this

        std::cout << "Cannot shuffle TRAIN_DATALOADER & TEST_DATALOADER since they have not been set yet. Put shuffle() after setting both dataloaders." << std::endl;
        return *this;
    }

    model.train_dl->shuffle();
    model.test_dl->shuffle();

    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::LINEAR(int in, int out)
{
    model.cardstack.push_back(std::move(std::make_unique<LinearCard>(in, out)));
    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::RELU()
{
    model.cardstack.push_back(std::move(std::make_unique<ReluCard>()));
    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::LEAKYRELU()
{
    model.cardstack.push_back(std::move(std::make_unique<LeakyReluCard>()));
    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::SOFTMAX()
{
    model.cardstack.push_back(std::move(std::make_unique<SoftmaxCard>()));
    return *this;
};

ml::ModelBuilder &ml::ModelBuilder::CCE()
{
    model.cardstack.push_back(std::move(std::make_unique<CCECard>()));
    return *this;
};

// --- THESE DONT EXIST YET, I NEED TO IMPLEMENT BIN CLASS and REGRESSION ---
// ml::ModelBuilder &ml::ModelBuilder::BCE()
// {
//     model.cardstack.push_back(std::move(std::make_unique<BCECard>()));
//     return *this;
// };

// ml::ModelBuilder &ml::ModelBuilder::RMSE()
// {
//     model.cardstack.push_back(std::move(std::make_unique<RMSECard>()));
//     return *this;
// };
// ---------------------------------------------------------------------------

// --- MODEL CODE ---

void ml::Model::test() {};

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
};

void ml::Model::do_forward_pass(Batch &b)
{

    for (auto &c : this->cardstack)
    {

        c->forward(b);
    }
};

void ml::Model::print_state()
{
    std::cout << "TOTAL EPOCHS: " << this->epoch << std::endl;
};

void ml::Model::do_backward_pass(Batch &b)
{

    for (int i = this->cardstack.size() - 1; i >= 0; i--)
    {
        this->cardstack[i]->backward(b);
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
};

void ml::Model::reset_grad(Batch &b)
{
    for (auto &c : this->cardstack)
    {
        if (c->has_weight())
        {
            c->reset_grad();
        };
    };

    b.batch_grad_prod.setZero();
    b.batch_loss.setZero();
};

void ml::Model::do_one_epoch(Batch &b)
{

    int num_inputs = 0;

    double current_loss = 0.0f;

    int cached_epoch_reached = this->EPOCH_REACHED;

    while (true)
    {
        this->train_dl->get_batch(b);

        if (b.all_batches_done)
        {
            break;
        };

        this->do_forward_pass(b);

        current_loss += b.batch_loss.sum();
        num_inputs += b.batch_loss.size();

        this->do_backward_pass(b);
        this->do_an_update();
        this->reset_grad(b);

        if (num_inputs % 100 == 0)
        {
            std::cout << "\rEPOCH [" << cached_epoch_reached << "] TRAINING: " << num_inputs << std::flush;
        };
    };

    this->EPOCH_TOTAL_CURRENT_LOSS = current_loss;
    this->EPOCH_AVG_CURRENT_LOSS = current_loss / num_inputs;
};

void ml::Model::do_one_test_pass()
{
    int total_inputs = 0;
    double current_loss_test = 0.0f;
    int correct_total_predictions = 0;

    Batch b_test;

    int cached_epoch_reached_test = this->EPOCH_REACHED;

    while (true)
    {
        int num_samples_in_batch_test = 0;

        this->test_dl->get_batch(b_test);

        if (b_test.all_batches_done)
        {
            break;
        }

        this->do_forward_pass(b_test);
        current_loss_test += b_test.batch_loss.sum();
        num_samples_in_batch_test = b_test.batch_sample.cols();

        total_inputs += num_samples_in_batch_test;

        if (total_inputs % 50 == 0)
        {
            std::cout << "\rEPOCH [" << cached_epoch_reached_test << "] TEST: " << total_inputs << std::flush;
        };

        for (int i = 0; i < num_samples_in_batch_test; ++i)
        {

            int max_pred_idx = 0;
            int true_label_idx = 0;

            b_test.batch_sample.col(i).maxCoeff(&max_pred_idx);
            b_test.batch_label.col(i).maxCoeff(&true_label_idx);

            if (max_pred_idx == true_label_idx)
            {
                correct_total_predictions++;
            }
        }
    }

    float accuracy = static_cast<float>(correct_total_predictions) / total_inputs * 100.0f;

    std::cout << "\t\tLOSS: " << current_loss_test / total_inputs;
    std::cout << "\tACCURACY: " << accuracy << "% (" << correct_total_predictions << "/" << total_inputs << ")" << std::endl;
    this->test_dl->reset(b_test);
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

        Batch b;
        this->do_one_epoch(b);
        std::cout << "\tLOSS: " << this->EPOCH_AVG_CURRENT_LOSS << std::endl;
        this->train_dl->reset(b);

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
}

void ml::Model::shape_check()
{
    Batch b;

    this->train_dl->get_batch(b);

    for (int i = 0; i < this->cardstack.size(); i++)
    {
        this->cardstack[i]->check_shape(b);
    }

    this->train_dl->reset(b);
}
