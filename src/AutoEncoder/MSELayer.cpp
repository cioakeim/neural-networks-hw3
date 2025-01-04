#include "AutoEncoder/MSELayer.hpp"


void MSELayer::forward(const PassContext& context){
  FeedForwardLayer::forward(context);
}


float MSELayer::loss(const PassContext& context){
  const MatrixXf& output=output_interface->forward_signal;
  return (context.input.array()-output.array()).array().pow(2).mean();
}


int MSELayer::prediction_success(const PassContext& context){
  std::cerr<<"MSE Layer can't predict anything.. Aborting.."<<std::endl;
  exit(1);
  return -1;
}


void MSELayer::backward(const PassContext& context){
  // Derivative in MSE is just y_bar-y 
  const MatrixXf& output=output_interface->forward_signal;
  MatrixXf error=output-context.input;
  const E::MatrixXf& in=(input_interface->type==Input)?
    context.input:input_interface->forward_signal;
  // Pass backward if needed
  if(input_interface->type!=Input){
    MatFunction func=input_interface->f_dot;
    input_interface->backward_signal=
      (weights.transpose()*error).cwiseProduct(func(in));
  } 
  if(!lockParams){
    updateWeights(in, error);
  }
}
