#include "MLP/ActivationFunctions.hpp"
#include <cmath>
#include <random>

E::MatrixXf reLU(const E::MatrixXf& in){
  return in.cwiseMax(0.0);
}

E::MatrixXf reLUder(const E::MatrixXf& reLU_output){
  return (reLU_output.array()>0).cast<float>();
}

E::MatrixXf leakyReLU(const E::MatrixXf& in,const float a){
  return in.cwiseMax(a*in);
}

E::MatrixXf leakyReLUder(const E::MatrixXf& output,const float a){
  const int rows=output.rows();
  const int cols=output.cols();

  return (output.array()>0).select(
    E::MatrixXf::Constant(rows,cols,1),
    E::MatrixXf::Constant(rows,cols,a)
  );
}


// Tanh
E::VectorXf tanh(const E::VectorXf& in){
  return in.array().tanh();
}

E::VectorXf tanhder(const E::VectorXf& tanh_output){
  return (1- tanh_output.array().square());
}


// Identity
E::MatrixXf linear(const E::MatrixXf& in){
  return in;
}

E::MatrixXf linearder(const E::MatrixXf& out){
  return E::MatrixXf::Ones(out.rows(),out.cols());
}
