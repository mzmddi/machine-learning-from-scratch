
// --- NOTES ---

// --- INCLUDES ---

#ifndef CARD_H
#define CARD_H

#include <iostream>
#include <Eigen/Core>

#include "Batch.h"

// --- CODE ---

namespace ml
{

    class Card
    {
    public:
        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC SHARED BY ALL ---

        Eigen::MatrixXf input_batch;

        virtual void print() = 0;

        virtual bool has_weight() { return false; }

        virtual void forward(Batch &b) = 0;
        virtual void backward(Batch &b) = 0;
        virtual void update(float lr) {};
        virtual void reset_grad() {};
        virtual void check_shape(Batch &b) {};
        void check_nan_inf_forward(Batch &b);
        void check_nan_inf_backward(Batch &b);

        virtual ~Card() {}
    };

    class LinearCard : public Card
    {
        int in, out;

        Eigen::MatrixXf weight;

        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC ---

        Eigen::MatrixXf gradient;

    public:
        LinearCard(int in, int out);

        void print() override;

        bool has_weight() override { return true; };

        void forward(Batch &b) override;

        void backward(Batch &b) override;

        void update(float lr) override;

        void reset_grad() override;

        void check_shape(Batch &b) override;

        ~LinearCard() override {};
    };

    class ReluCard : public Card
    {

        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC ---

    public:
        ReluCard();

        void print() override;

        void forward(Batch &b) override;

        void backward(Batch &b) override;

        void check_shape(Batch &b) override;

        ~ReluCard() override {}
    };

    class LeakyReluCard : public Card
    {

        float alpha_for_leak = 0.01f;

        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC ---

    public:
        LeakyReluCard();
        void print() override;
        void forward(Batch &b) override;

        void backward(Batch &b) override;
        ~LeakyReluCard() override {}
    };

    // class FlattenCard : public Card
    // {
    // public:
    //     FlattenCard();
    //     void print() override;
    //     void forward() override;
    //     ~FlattenCard() override {}
    // };

    // class Conv1dCard : public Card
    // {
    // public:
    //     Conv1dCard();
    //     void print() override;
    //     void forward() override;
    //     ~Conv1dCard() override {}
    // };

    // class Conv2dCard : public Card
    // {
    // public:
    //     Conv2dCard();
    //     void print() override;
    //     void forward() override;
    //     ~Conv2dCard() override {}
    // };

    class SoftmaxCard : public Card
    {

    public:
        SoftmaxCard();
        void print() override;
        void forward(Batch &b) override;

        void backward(Batch &b) override;
        void check_shape(Batch &b) override;
        ~SoftmaxCard() override {}
    };

    // class SigmoidCard : public Card
    // {
    // public:
    //     SigmoidCard();
    //     void print() override;
    //     void forward() override;
    //     ~SigmoidCard() override {}
    // };

    class CCECard : public Card
    {
    public:
        CCECard() {};
        void print() override;
        void forward(Batch &b) override;
        void backward(Batch &b) override;
        void check_shape(Batch &b) override;
        ~CCECard() override {};
    };

    // class MSECard : public Card
    // {
    // }

    // class BCECard : public Card
    // {
    // }

};

#endif