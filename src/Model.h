
// --- NOTES ---

// --- INCLUDES ---

#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>
#include <string>

#include "Card.h"
#include "Dataloader.h"
#include "Optimizer.h"

#include "Batch.h"

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

        float learning_rate = 1e-1f;

        int test_interval = 10;

        // --- TEMPORARY VALUES FOR INTERNAL LOGIC ---

        int EPOCH_COUNTER = epoch;
        int EPOCH_REACHED = 1;
        double EPOCH_TOTAL_CURRENT_LOSS = 0.0f;
        double EPOCH_AVG_CURRENT_LOSS = 0.0f;

        Model() {};
        // constructor is private
        // because i want the user to use the ModelBuilder class
        // to build the model object

    public:
        void test();
        /*
        void set_loss(std::unique_ptr<Loss> l);
        void set_optmizer(std::unique_ptr<Opto> o);
        NEED TO IMPLEMENT THEM IN THE FUTURE
        FUTURE PLANNING BASICALLY
        */
        void print_architecture();
        void train();

        void normalize_datasets()
        {
            this->test_dl->normalize();
            this->train_dl->normalize();
        };

        void shape_check();

    private:
        friend class ModelBuilder;

        void print_state();
        void do_forward_pass(Batch &b);
        void do_one_epoch(Batch &b);
        void do_backward_pass(Batch &b);
        void do_an_update();
        void reset_grad(Batch &b);
        void do_one_test_pass();
        void check_gates(std::vector<std::string> gates);
        void check_all_gates();
    };

    class ModelBuilder
    {
    private:
        ml::Model model;

    public:
        ModelBuilder &train_dataloader_filename(std::string filename);
        ModelBuilder &test_dataloader_filename(std::string filename);

        ModelBuilder &train_dataloader_filename(std::string filename, int batch_size);
        ModelBuilder &test_dataloader_filename(std::string filename, int batch_size);

        ModelBuilder &learning_rate(float lr);
        ModelBuilder &test_interval(int ti);
        ModelBuilder &epoch(int e);

        ModelBuilder &shuffle();

        ModelBuilder &LINEAR(int in, int out);
        ModelBuilder &RELU();
        ModelBuilder &LEAKYRELU();
        ModelBuilder &SOFTMAX();

        ModelBuilder &CCE();
        // --- they dont exist yet, i need to implement the cards first
        // ModelBuilder &BCE();
        // ModelBuilder &RMSE();

        // ModelBuilder &SGD_opt();
        // ModelBuilder &MOMENTUM_opt();
        // ModelBuilder &ADAM_opt();

        operator Model() { return std::move(model); };
    };
};

#endif