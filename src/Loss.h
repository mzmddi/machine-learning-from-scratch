#ifndef LOSS_H
#define LOSS_H

#include "Sample.h"

class Loss
{

public:
    virtual float compute_loss(Sample &s);
};

class CCE : public Loss
{
public:
    float compute_loss(Sample &s) override;
};

#endif