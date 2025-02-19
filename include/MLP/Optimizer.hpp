#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#endif
#include <Eigen/Dense>
#include <iostream>

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;


enum OptimizerType{SGD,Adam};


struct SGDConfig{
  float rate;
};

struct AdamConfig{
  float rate;
  float beta_1;
  float beta_2;
  int mat_rows,mat_cols;
};


struct OptimizerConfig{
  OptimizerType type;
  SGDConfig sgd;
  AdamConfig adam;
};

/**
 * @brief Optimizer class for MLP
  */
class Optimizer{
private:
  // Universal 
  float rate;
  OptimizerType type;
  // For adam
  MatrixXf m;
  MatrixXf u;
  float epsilon;
  float beta_1,beta_2;
  float beta_1t,beta_2t;

public:
  Optimizer(){};

  void setRate(const float rate){
    std::cout<<"New rate: "<<rate<<std::endl;
    this->rate=rate;
  }

  void setAdam(AdamConfig config);
               
  void setSGD(SGDConfig config);

  void configure(OptimizerConfig config,const MatrixXf& mat);

  void update(const MatrixXf& mat_gradients,
              MatrixXf& mat);
};

#endif
