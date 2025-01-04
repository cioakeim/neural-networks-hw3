#include "MLP/Optimizer.hpp"
#include <iostream>

#define EPSILON 1e-8
#define DECAY 1e-8

void Optimizer::setAdam(AdamConfig config){
  type=Adam;
  this->rate=config.rate;
  this->epsilon=EPSILON;
  this->beta_1=this->beta_1t=config.beta_1;
  this->beta_2=this->beta_2t=config.beta_2;
  this->batch_size=config.batch_size;

  m=MatrixXf(config.mat_rows,config.mat_cols).setZero();
  u=MatrixXf(config.mat_rows,config.mat_cols).setZero();
}


void Optimizer::setSGD(SGDConfig config){
  type=SGD;
  this->rate=config.rate;
}


void Optimizer::configure(OptimizerConfig config,const MatrixXf& mat){
  switch(config.type){
  case SGD:
    setSGD(config.sgd);
    break;
  case Adam:
    config.adam.mat_rows=mat.rows();
    config.adam.mat_cols=mat.cols();
    setAdam(config.adam);
    break;
  }
}


void Optimizer::update(const MatrixXf& mat_gradients,
                       MatrixXf& mat){
  switch(type){
  case Adam:
    // Moment calculation
    m=beta_1*m+(1-beta_1)*mat_gradients;
    u.array()=beta_2*u.array()+
                (1-beta_2)*mat_gradients.array().square()*batch_size;

    // Calculate weights
    mat.array()-=(rate/(1-beta_1t))*
      (m.array()/((u.array()/(1-beta_2t)).sqrt()+epsilon));

    // Diminish beta_t
    beta_1t*=beta_1t;
    beta_2t*=beta_2t;
    break;
  case SGD:
    //std::cout<<"Gradients div: "<<mat_gradients.array().pow(2).mean()<<std::endl;
    mat-=(rate)*mat_gradients+DECAY*mat;
    break;
  }
}


