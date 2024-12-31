#include "AutoEncoder/MSELayer.hpp"


void MSELayer::forward(const PassContext& context){
  const MatrixXf& input=(previous==nullptr)?context.input:previous->outputRef();
  // NO NON-LINEARITY
  output=(weights*input).colwise()+biases.col(0);
}


float MSELayer::loss(const PassContext& context){
  return (context.input.array()-output.array()).array().pow(2).mean();
}


int MSELayer::prediction_success(const PassContext& context){
  std::cerr<<"MSE Layer can't predict anything.. Aborting.."<<std::endl;
  exit(1);
  return -1;
}


void MSELayer::backward(const PassContext& context){
  // Derivative in MSE is just y_bar-y 
  if(next!=nullptr){
    std::cerr<<"MSE Layer can't be hidden"<<std::endl;
    exit(1);
  }
  MatrixXf error=output-context.input;
  const E::MatrixXf& in=(previous==nullptr)?context.input:previous->outputRef();
  // Pass backward if needed
  if(previous!=nullptr){
    MatFunction func=previous->getFDot();
    input_error=(weights.transpose()*error).cwiseProduct(func(in));
  } 
  if(!lockWeights){
    updateWeights(in, error);
  }
}
