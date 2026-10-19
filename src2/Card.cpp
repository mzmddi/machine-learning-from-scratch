// --- NOTES ---

// --- INCLUDES ---

#include <iostream>
#include <Eigen/Core>
#include <random>

#include "Card.h"
#include "Sample.h"

// --- CODE ---

ml::LinearCard::LinearCard(int in, int out)
{

    this->weight = Eigen::MatrixXf::Zero(out, in + 1);

    std::random_device rd;
    std::mt19937 gen(rd());

    float stddev = std::sqrt(2.0f / static_cast<float>(in));
    std::normal_distribution<float> d(0.0f, stddev);

    for (int r = 0; r < out; ++r)
    {
        for (int c = 0; c < in; ++c)
        // in is used instead of in+1 because we want the bias column to be all 0
        {
            this->weight(r, c) = d(gen);
        };
    };

    this->gradient = Eigen::MatrixXf::Zero(out, in + 1);

    this->in = in;
    this->out = out;
};

void ml::LinearCard::print()
{
    std::cout << "LinearCard " << in << "+1 " << out << std::endl;
};

void ml::LinearCard::forward(Sample &s)
{

    Eigen::VectorXf m(s.features.size() + 1);

    m << s.features, 1.0f;

    this->input_vector = m;

    Eigen::VectorXf output = this->weight * m;

    s.features = output;
}

void ml::LinearCard::backward(Sample &s)
{
    //  shapes => (out, 1) * (1, in+1)
    //  => (out, in+1) exactly the same shape as the weight which is what we want

    this->gradient = s.grad_prod * this->input_vector.transpose();

    // matrix without bias => (out, in)
    // error from layer above => (out, 1) its a column vector
    // W.T * error => (in,out) * (out, 1)
    // (in, 1) which in here is basically the out of the next layer in the backprop

    Eigen::MatrixXf weight_without_bias = this->weight.leftCols(this->in);

    Eigen::VectorXf current_error = weight_without_bias.transpose() * s.grad_prod;

    s.grad_prod = current_error;
};

void ml::LinearCard::update(float lr)
{
    this->weight = this->weight - (lr * this->gradient);
}

void ml::LinearCard::reset_grad()
{
    this->gradient.setZero();
}

ml::ReluCard::ReluCard() {};

void ml::ReluCard::print()
{
    std::cout << "ReluCard" << std::endl;
};

void ml::ReluCard::forward(Sample &s)
{
    this->input_vector = s.features;

    s.features = s.features.cwiseMax(0.0f);
}

void ml::ReluCard::backward(Sample &s)
{

    Eigen::VectorXf relu_derivative = (this->input_vector.array() > 0.0f).cast<float>();

    Eigen::VectorXf d_prev = s.grad_prod.cwiseProduct(relu_derivative);

    s.grad_prod = d_prev;
};

ml::LeakyReluCard::LeakyReluCard() {};

void ml::LeakyReluCard::print()
{
    std::cout << "LeakyReluCard" << std::endl;
}

void ml::LeakyReluCard::forward(Sample &s)
{
    this->input_vector = s.features;

    float a = this->alpha_for_leak;

    s.features = s.features.unaryExpr([a](float val)
                                      { return val > 0.0f ? val : val * a; });
};

void ml::LeakyReluCard::backward(Sample &s) { std::cout << "leakyRelu back" << std::endl; };

// ml::FlattenCard::FlattenCard() {};

// void ml::FlattenCard::print()
// {
//     std::cout << "FlattenCard" << std::endl;
// }

// void ml::FlattenCard::forward() {}

// ml::Conv1dCard::Conv1dCard() {};

// void ml::Conv1dCard::print()
// {
//     std::cout << "Conv1dCard" << std::endl;
// }

// void ml::Conv1dCard::forward() {}

// ml::Conv2dCard::Conv2dCard() {};

// void ml::Conv2dCard::print()
// {
//     std::cout << "Conv2dCard" << std::endl;
// }

// void ml::Conv2dCard::forward() {}

ml::SoftmaxCard::SoftmaxCard() {};

void ml::SoftmaxCard::print()
{
    std::cout << "SoftmaxCard" << std::endl;
}

void ml::SoftmaxCard::forward(Sample &s)
{
    this->input_vector = s.features;

    float max_val = s.features.maxCoeff();

    Eigen::VectorXf expv = (s.features.array() - max_val).exp();

    s.features = (expv / expv.sum()); //.cwiseMax(1e-10f);
}

void ml::SoftmaxCard::backward(Sample &s)
{

    // when paired with CCE the equation collapses into just prediction - label

    Eigen::VectorXf prod = s.features - s.label;

    s.grad_prod = prod;
};

// ml::SigmoidCard::SigmoidCard() {};

// void ml::SigmoidCard::print()
// {
//     std::cout << "SigmoidCard" << std::endl;
// }

// void ml::SigmoidCard::forward() {}

void ml::CCECard::print()
{
    std::cout << "CCE" << std::endl;
}

void ml::CCECard::forward(Sample &s)
{
    const float epsi = 1e-15f;

    this->input_vector = s.features;

    Eigen::VectorXf clipped_preds = s.features.cwiseMax(epsi).cwiseMin(1.0f - epsi);
    Eigen::VectorXf log_preds = clipped_preds.array().log();
    s.loss = -(s.label.dot(log_preds));
}

void ml::CCECard::backward(Sample &s)
{
    // since we are starting the product at the softmax layer for the CCE, we just skip this and do nothing
}