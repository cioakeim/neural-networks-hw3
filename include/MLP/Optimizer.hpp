#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <Eigen/Dense>

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;


enum OptimizerMode{SGD,Adam};

struct SGDConfig{
  float rate;
  int batch_size;
};

struct AdamConfig{
  float rate;
  int batch_size;
  float beta_1;
  float beta_2;
  int mat_rows,mat_cols;
};

/**
 * @brief Optimizer class for MLP
  */
class Optimizer{
private:
  // Universal 
  float rate;
  OptimizerMode mode;
  // For adam
  MatrixXf m;
  MatrixXf u;
  float epsilon;
  float beta_1,beta_2;
  float beta_1t,beta_2t;

public:
  Optimizer(){};

  void setAdam(AdamConfig config);
               
  void setSGD(SGDConfig config);

  void update(const MatrixXf& mat_gradients,
              MatrixXf& mat);
};

#endif
