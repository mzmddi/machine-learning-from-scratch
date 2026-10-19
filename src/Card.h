
// --- NOTES ---

// --- INCLUDES ---

#ifndef CARD_H
#define CARD_H

#include <iostream>
#include <Eigen/Core>

#include "Sample.h"

// --- CODE ---

namespace ml
{

    class Card
    {
    public:
        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC SHARED BY ALL ---

        Eigen::VectorXf input_vector;

        virtual void print() = 0;

        virtual bool has_weight() { return false; }

        virtual void forward(Sample &s) = 0;
        virtual void backward(Sample &s) = 0;
        virtual void update(float lr) {};
        virtual void reset_grad() {};

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

        void forward(Sample &s) override;

        void backward(Sample &s) override;

        void update(float lr) override;

        void reset_grad() override;

        ~LinearCard() override {};
    };

    class ReluCard : public Card
    {

        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC ---

    public:
        ReluCard();

        void print() override;

        void forward(Sample &s) override;
        void backward(Sample &s) override;

        ~ReluCard() override {}
    };

    class LeakyReluCard : public Card
    {

        float alpha_for_leak = 0.01f;

        // --- TEMPORARY VALUES FOR THE INTERNAL LOGIC ---

    public:
        LeakyReluCard();
        void print() override;
        void forward(Sample &s) override;
        void backward(Sample &s) override;
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
        void forward(Sample &s) override;
        void backward(Sample &s) override;
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
        void forward(Sample &s) override;
        void backward(Sample &s) override;
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