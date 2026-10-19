
// --- NOTES ---

// --- INCLUDES ---

#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>
#include <string>

#include "Card.h"
#include "Dataloader.h"
#include "Sample.h"

// --- CODE ---

namespace ml
{
    class Model
    {

        std::vector<std::unique_ptr<Card>> cardstack;

        std::unique_ptr<DataLoader> test_dl;
        std::unique_ptr<DataLoader> train_dl;

        std::unordered_map<std::string, bool> gates = {
            {"GATE_TEST_DL", false},
            {"GATE_TRAIN_DL", false},
            {"GATE_LOSS_SET", false},
            {"GATE_CARD_SHAPES_COMPATIBLE", false},
            {"GATE_ARCHITECTURE_STRUCTURE", false},
            {"GATE_OPTMIZER_SET", false},
            {"GATE_OBJECTIVE_FUNC_MATCHES", false}};

        int epoch = 10;

        float learning_rate = 1e-2f;

        int test_interval = 2;

        // --- TEMPORARY VALUES FOR INTERNAL LOGIC ---

        int EPOCH_COUNTER = epoch;
        int EPOCH_REACHED = 1;
        double EPOCH_TOTAL_CURRENT_LOSS = 0.0f;
        double EPOCH_AVG_CURRENT_LOSS = 0.0f;

    public:
        Model() {};
        void test();
        void add(std::unique_ptr<Card> c);
        /*
        void set_loss(std::unique_ptr<Loss> l);
        void set_optmizer(std::unique_ptr<Opto> o);
        NEED TO IMPLEMENT THEM IN THE FUTURE
        FUTURE PLANNING BASICALLY
        */
        void print_architecture();
        void train();

        // --- SETTERS ---
        void set_test_dataloader(std::unique_ptr<DataLoader> dl);
        void set_train_dataloader(std::unique_ptr<DataLoader> dl);
        void set_epochs(int s)
        {
            this->epoch = s;
            this->EPOCH_COUNTER = s;
        };
        void set_learning_rate(float lr) { this->learning_rate = lr; };
        void set_test_interval(int s) { this->test_interval = s; };

    private:
        void print_state();
        void do_forward_pass(Sample &s);
        void do_one_epoch(Sample &s);
        void do_backward_pass(Sample &s);
        void do_an_update();
        void do_a_grad_reset();
        void do_one_test_pass();
        void check_gates(std::vector<std::string> gates);
        void check_all_gates();
    };

};

#endif