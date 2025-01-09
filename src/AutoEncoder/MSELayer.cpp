#include "AutoEncoder/MSELayer.hpp"
#include "MLP/ActivationFunctions.hpp"

void MSELayer::configure(LayerConfig config){
  FeedForwardLayer::configure(config);
  output_interface->f=linear;
  output_interface->f_dot=linearder;
}


void MSELayer::forward(const PassContext& context){
  FeedForwardLayer::forward(context);
}


float MSELayer::loss(const PassContext& context){
  const MatrixXf& output=output_interface->forward_signal;
  const MatrixXf& error=(context.input.array()-output.array()).array().pow(2);
  return error.mean()*error.cols();
}


int MSELayer::prediction_success(const PassContext& context){
  return -1;
}


void MSELayer::backward(const PassContext& context){
  const float lambda=2;
  // Derivative in MSE is just y_bar-y 
  const MatrixXf& output=output_interface->forward_signal;
  MatrixXf error=(lambda/output.rows())*(output-context.input);
  const E::MatrixXf& in=(input_interface->type==Input)?
    context.input:input_interface->forward_signal;
  // Pass backward if needed
  if(input_interface->type!=Input){
    MatFunction func=input_interface->f_dot;
    if(batch_normalization){
      float c=norm.scale(1,0)/sqrt(norm.var+EPSILON);
      input_interface->backward_signal=
        (weights.transpose()*(c*error)).cwiseProduct(func(in));
      if(!lockParams)
        norm.update(error);
    }
    else{
      input_interface->backward_signal=
        (weights.transpose()*error).cwiseProduct(func(in));
    }
  } 
  if(!lockParams){
    updateWeights(in, error);
  }
}
