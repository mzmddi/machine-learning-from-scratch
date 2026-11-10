// --- NOTES ---

// --- INCLUDES ---

#include <iostream>
#include <Eigen/Core>
#include <random>

#include "Card.h"
#include "Batch.h"

// --- CODE ---

void ml::Card::check_nan_inf_forward(Batch &b)
{
    if (b.batch_sample.hasNaN() || !b.batch_sample.allFinite())
    {
        std::cout << "FORWARD Theres a nan or an infinite somewhere in this layer batch sample: ";
        this->print();

        std::exit(0);
    }

    if (b.batch_grad_prod.hasNaN() || !b.batch_grad_prod.allFinite())
    {
        std::cout << "FORWARD Theres a nan or an infinite somewhere in this layer batch sample: ";
        this->print();

        std::exit(0);
    }

    if (b.batch_loss.hasNaN() || !b.batch_loss.allFinite())
    {
        std::cout << "FORWARD Theres a nan or an infinite somewhere in this layer batch sample: ";
        this->print();

        std::exit(0);
    }
}

void ml::Card::check_nan_inf_backward(Batch &b)
{
    if (b.batch_sample.hasNaN() || !b.batch_sample.allFinite())
    {
        std::cout << "BACKWARD Theres a nan or an infinite somewhere in this layer batch sample: ";
        this->print();

        std::exit(0);
    }

    if (b.batch_grad_prod.hasNaN() || !b.batch_grad_prod.allFinite())
    {
        std::cout << "BACKWARD Theres a nan or an infinite somewhere in this layer batch sample: ";
        this->print();

        std::exit(0);
    }

    if (b.batch_loss.hasNaN() || !b.batch_loss.allFinite())
    {
        std::cout << "BACKWARD Theres a nan or an infinite somewhere in this layer batch sample: ";
        this->print();

        std::exit(0);
    }
}

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

void ml::LinearCard::forward(Batch &b)
{

    int num_of_features = b.batch_sample.rows();
    int num_of_samples = b.batch_sample.cols();

    Eigen::MatrixXf m(num_of_features + 1, num_of_samples);

    m.topRows(num_of_features) = b.batch_sample;

    m.bottomRows(1).setOnes();

    this->input_batch = m;

    Eigen::MatrixXf output = this->weight * m;

    b.batch_sample = output;

    this->check_nan_inf_forward(b);
}

void ml::LinearCard::backward(Batch &b)
{
    //  shapes => (out, 1) * (1, in+1)
    //  => (out, in+1) exactly the same shape as the weight which is what we want

    this->gradient = (b.batch_grad_prod * this->input_batch.transpose()) / static_cast<float>(b.batch_sample.size());
    // need to divide it all by the size because otherwise the weights are going to explode in value

    // matrix without bias => (out, in)
    // error from layer above => (out, 1) its a column vector
    // W.T * error => (in,out) * (out, 1)
    // (in, 1) which in here is basically the out of the next layer in the backprop

    Eigen::MatrixXf weight_without_bias = this->weight.leftCols(this->in);

    Eigen::MatrixXf current_error = weight_without_bias.transpose() * b.batch_grad_prod;

    b.batch_grad_prod = current_error;

    this->check_nan_inf_backward(b);
};

void ml::LinearCard::update(float lr)
{
    if (this->gradient.hasNaN() || !this->gradient.allFinite())
    {
        std::cout << "\nTheres a nan or an infinite somewhere in this layer GRADIENT: ";
        this->print();

        std::exit(0);
    }

    if (this->weight.hasNaN() || !this->weight.allFinite())
    {
        std::cout << "Theres a nan or an infinite somewhere in this layer WEIGHT: ";
        this->print();

        std::exit(0);
    }

    this->weight -= lr * this->gradient;
}

void ml::LinearCard::reset_grad()
{
    this->gradient.setZero();
}

void ml::LinearCard::check_shape(Batch &b)
{

    int input_weigth_rows = this->weight.rows();
    int input_weight_cols = this->weight.cols();

    int input_batch_rows = b.batch_sample.rows();
    int input_batch_cols = b.batch_sample.cols();

    this->forward(b);

    int output_batch_rows = b.batch_sample.rows();
    int output_batch_cols = b.batch_sample.cols();

    std::cout << "LINEAR FORWARD \t\t (" << input_weigth_rows << ", " << input_weight_cols << ") * (" << input_batch_rows << ", " << input_batch_cols << ") = (" << output_batch_rows << ", " << output_batch_cols << ")" << std::endl;

    this->reset_grad();
};

ml::ReluCard::ReluCard() {};

void ml::ReluCard::print()
{
    std::cout << "ReluCard" << std::endl;
};

void ml::ReluCard::forward(Batch &b)
{
    this->input_batch = b.batch_sample;

    b.batch_sample = b.batch_sample.cwiseMax(0.0f);

    this->check_nan_inf_forward(b);
}

void ml::ReluCard::backward(Batch &b)
{

    Eigen::MatrixXf relu_derivative = (this->input_batch.array() > 0.0f).cast<float>();

    Eigen::MatrixXf d_prev = b.batch_grad_prod.cwiseProduct(relu_derivative);

    b.batch_grad_prod = d_prev;
    this->check_nan_inf_backward(b);
};

void ml::ReluCard::check_shape(Batch &b)
{

    int input_batch_rows = b.batch_sample.rows();
    int input_batch_cols = b.batch_sample.cols();

    this->forward(b);

    int output_batch_rows = b.batch_sample.rows();
    int output_batch_cols = b.batch_sample.cols();

    std::cout << "RELU FORWARD \t\t (" << input_batch_rows << ", " << input_batch_cols << ") = (" << output_batch_rows << ", " << output_batch_cols << ")" << std::endl;
};

ml::LeakyReluCard::LeakyReluCard() {};

void ml::LeakyReluCard::print()
{
    std::cout << "LeakyReluCard" << std::endl;
}

void ml::LeakyReluCard::forward(Batch &b)
{
    this->input_batch = b.batch_sample;

    float a = this->alpha_for_leak;

    b.batch_sample = b.batch_sample.unaryExpr([a](float val)
                                              { return val > 0.0f ? val : val * a; });

    this->check_nan_inf_forward(b);
};

void ml::LeakyReluCard::backward(Batch &b) { std::cout << "leakyRelu back" << std::endl; };

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

void ml::SoftmaxCard::forward(Batch &b)
{
    this->input_batch = b.batch_sample;

    int num_of_samples = b.batch_sample.cols();

    for (int i = 0; i < num_of_samples; i++)
    {
        Eigen::VectorXf col = b.batch_sample.col(i);

        float max_val = col.maxCoeff();

        Eigen::VectorXf expv = (col.array() - max_val).exp();

        float sum_exp = expv.sum();

        if (sum_exp == 0.0f)
            sum_exp = 1e-15f;
        // safety check ?

        b.batch_sample.col(i) = expv / sum_exp;
    }

    this->check_nan_inf_forward(b);
}

void ml::SoftmaxCard::backward(Batch &b)
{

    // when paired with CCE the equation collapses into just prediction - label

    Eigen::MatrixXf prod = b.batch_sample - b.batch_label;

    b.batch_grad_prod = prod;
    this->check_nan_inf_backward(b);
};

void ml::SoftmaxCard::check_shape(Batch &b)
{

    int input_batch_rows = b.batch_sample.rows();
    int input_batch_cols = b.batch_sample.cols();

    this->forward(b);

    int output_batch_rows = b.batch_sample.rows();
    int output_batch_cols = b.batch_sample.cols();

    std::cout << "SOFTMAX FORWARD \t (" << input_batch_rows << ", " << input_batch_cols << ") = (" << output_batch_rows << ", " << output_batch_cols << ")" << std::endl;
}

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

void ml::CCECard::forward(Batch &b)
{
    const float epsi = 1e-15f;

    this->input_batch = b.batch_sample;

    int num_of_samples = b.batch_sample.cols();

    Eigen::MatrixXf clipped_preds = b.batch_sample.cwiseMax(epsi).cwiseMin(1.0f - epsi);

    b.batch_loss.resize(num_of_samples);
    b.batch_loss.setZero();

    for (int i = 0; i < num_of_samples; i++)
    {

        auto c = clipped_preds.col(i);
        auto one_label = b.batch_label.col(i);

        Eigen::VectorXf log_pred_i = c.array().log();

        b.batch_loss(i) = -(one_label.cwiseProduct(log_pred_i)).sum();
    }

    this->check_nan_inf_forward(b);
}

void ml::CCECard::backward(Batch &b)
{
    // since we are starting the product at the softmax layer for the CCE, we just skip this and do nothing
    this->check_nan_inf_backward(b);
}

void ml::CCECard::check_shape(Batch &b)
{

    int input_batch_rows = b.batch_sample.rows();
    int input_batch_cols = b.batch_sample.cols();

    int batch_label_rows = b.batch_label.rows();
    int batch_label_cols = b.batch_label.cols();

    std::cout << "CCE FORWARD \t\t labels: (" << batch_label_rows << ", " << batch_label_cols << ")\tinput batch: (" << input_batch_rows << ", " << input_batch_cols << ")";

    this->forward(b);

    int output_batch_rows = b.batch_sample.rows();
    int output_batch_cols = b.batch_sample.cols();

    std::cout << "\toutput batch: (" << output_batch_rows << ", " << output_batch_cols << ")" << std::endl;
}