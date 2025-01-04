#include "MLP/FeedForwardLayer.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <memory>
#include <random>
#include <iostream>
#include <fstream>


// Initialization
void FeedForwardLayer::configure(LayerConfig config){
  input_interface=config.input_interface;

  // Allocate matrices
  int input_sz,output_sz;
  input_sz=config.input_interface->size;
  output_sz=config.ff_config.output_sz;
  std::cout<<"Init dimension: "<<input_sz<<" "<<output_sz<<std::endl;
  weights=MatrixXf(output_sz,input_sz);
  biases=MatrixXf(output_sz,1);

  // Init optimizers
  weights_opt.configure(config.opt_config,weights);
  biases_opt.configure(config.opt_config,biases);

  // Define output interface
  output_interface=std::make_shared<LayerInterface>();
  output_interface->height=output_interface->size=output_sz;
  output_interface->width=output_interface->channels=1;
  output_interface->f=config.f;
  output_interface->f_dot=config.f_dot;
  init();
}


void FeedForwardLayer::init(){
  const int rows=weights.rows();
  const int cols=weights.cols();
  const float stddev= std::sqrt(2.0f/rows);
  // Init rng 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0,stddev);

  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      weights(i,j)=(dist(gen));
    }
    biases(i,0)=(dist(gen));
  }
}


void FeedForwardLayer::forward(const PassContext& context){
  const MatrixXf& input=(input_interface->type==Input)?
    context.input:input_interface->forward_signal;
  MatFunction f=output_interface->f;
  output_interface->forward_signal=f((weights*input).colwise()+biases.col(0));
}


float FeedForwardLayer::loss(const PassContext& context){
  MatrixXf loss=output_interface->forward_signal;
  #pragma omp parallel for 
  for(int i=0;i<loss.cols();i++){
    loss(context.labels(i),i)--; 
  }
  return loss.colwise().mean().sum();   
}


int FeedForwardLayer::prediction_success(const PassContext& context){
  int success_count=0;
  for(int i=0;i<context.labels.size();i++){
    E::MatrixXf::Index idx;
    output_interface->forward_signal.col(i).maxCoeff(&idx);
    success_count+=(context.labels(i)==idx);
  }
  return success_count;
}


// Backward
void FeedForwardLayer::backward(const PassContext& context){
  if(output_interface->type==Output){
    std::cerr<<"Base feedforward can't be output layer, can't backprop.."<<std::endl;
    exit(1);
  }
  const E::MatrixXf& error=output_interface->backward_signal;
  const E::MatrixXf& in=(input_interface->type==Input)?
    context.input:input_interface->forward_signal;
  if(input_interface->type!=Input){
    MatFunction func=input_interface->f_dot;
    input_interface->backward_signal=
      (weights.transpose()*error).cwiseProduct(func(in));
  }
  if(!lockWeights)
    updateWeights(in,error);
}

// For updating
void FeedForwardLayer::updateWeights(const MatrixXf& input,
                                     const MatrixXf& error){
  /*
  std::cout<<"Input: "<<input.rows()<<" "<<input.cols()<<std::endl;
  std::cout<<"Error: "<<error.rows()<<" "<<error.cols()<<std::endl;
  */
  MatrixXf weight_grads=error*input.transpose()/input.cols();
  MatrixXf bias_grads=error.rowwise().sum()/input.cols();
  weights_opt.update(weight_grads, weights);
  biases_opt.update(bias_grads, biases);
}


// I/O
void FeedForwardLayer::store(){
  ensure_a_path_exists(store_path);
  storeMatrixToFile(store_path+"/weights.csv",
                    weights);
  storeMatrixToFile(store_path+"/biases.csv",
                    biases);
}


void FeedForwardLayer::load(){
  ensure_a_path_exists(store_path);
  weights=loadMatrixFromFile(store_path+"/weights.csv");
  biases=loadMatrixFromFile(store_path+"/biases.csv");
}
