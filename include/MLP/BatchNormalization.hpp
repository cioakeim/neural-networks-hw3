#ifndef BATCH_NORM_HPP
#define BATCH_NORM_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#endif
#include <Eigen/Dense>
#include <MLP/Optimizer.hpp>

#define EPSILON 1e-10
#define ALPHA 0.9

namespace E=Eigen;

using E::MatrixXf;


/**
 * @brief Structure used for batch normalization
  */
struct BatchNormHandler{
  MatrixXf u_norm; //< Normalized activations
  // Parameters used for scaling (2x1 0 is mean 1 is var)
  MatrixXf scale;
  // Optimizer for the 2 parameters
  Optimizer opt;
  // Continusly update moments for inference
  float mean;
  float var;

  void init(OptimizerConfig opt_config);

  void setInferenceMoments(float mean,float var){
    this->mean=mean;this->var=var;
  }

  void setRate(float rate){
    opt.setRate(rate/10);
  }

  // Normalization either in training or inference
  void normalize(const MatrixXf& activation,
                 bool is_training);

  void update(const MatrixXf& error);
};


#endif
