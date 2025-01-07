#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#endif
#include <Eigen/Dense>

namespace E=Eigen;

// Each function comes with its derivative 

// ReLU

E::MatrixXf reLU(const E::MatrixXf& in);
E::MatrixXf reLUder(const E::MatrixXf& reLU_output);


E::MatrixXf leakyReLU(const E::MatrixXf& in,const float a);
E::MatrixXf leakyReLUder(const E::MatrixXf& in,const float a);

// Tanh
E::VectorXf tanh(const E::VectorXf& in);
E::VectorXf tanhder(const E::VectorXf& tanh_output);

E::MatrixXf linear(const E::MatrixXf& in);
E::MatrixXf linearder(const E::MatrixXf& out);



#endif
