#include "MLP/FeedForwardLayer.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <random>
#include <iostream>
#include <fstream>


// Initialization
void FeedForwardLayer::configure(LayerConfig config){
  // Allocate matrices
  int input_sz,output_sz,batch_size;
  if(previous==nullptr){
    std::cout<<"FIRST LAYERS"<<std::endl;
    input_sz=config.input_size;
  }
  else{
    InterfaceParams param=previous->outputParamRef();
    input_sz=param.height*param.width*param.channels;
  }
  output_sz=config.ff_config.feedforward_output;
  batch_size=config.batch_size;


  std::cout<<"Init dimension: "<<input_sz<<" "<<output_sz<<std::endl;
  weights=MatrixXf(output_sz,input_sz);
  biases=MatrixXf(output_sz,1);
  output=MatrixXf(output_sz,batch_size);
  input_error=MatrixXf(input_sz,batch_size);
  f=config.f;
  f_dot=config.f_dot;
  // Init optimizers
  switch(config.optimizer_mode){
  case SGD:
    config.sgd_config.batch_size=config.batch_size;
    weights_opt.setSGD(config.sgd_config);
    biases_opt.setSGD(config.sgd_config);
    break;
  case Adam:
    config.adam_config.batch_size=config.batch_size;
    config.adam_config.mat_rows=output_sz;
    config.adam_config.mat_cols=input_sz;
    weights_opt.setAdam(config.adam_config);
    config.adam_config.mat_cols=1;
    biases_opt.setAdam(config.adam_config);
    break;
  }
  // Set self out params
  output_param.height=output_sz;
  output_param.width=output_param.channels=1;
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


// Forward (in case of manual input or of previous one's)
void FeedForwardLayer::forward(const MatrixXf& input){
  output=f((weights*input).colwise()+biases.col(0));
  //std::cout<<"Out div: "<< output.array().sqrt().mean()<<std::endl;
}


void FeedForwardLayer::forward(){
  const MatrixXf& input=previous->outputRef();
  forward(input);
}


float FeedForwardLayer::loss(const VectorXi& labels){
  std::cerr<<"Feedforward CAN'T RETURN LOSS.."<<std::endl;
  exit(1);
}


// Backward (same idea as above)
void FeedForwardLayer::backward(const MatrixXf& input,
                                const VectorXi& labels){
  std::cerr<<"Feedforward CAN'T INIT ERRORS.."<<std::endl;
  exit(1);
}


void FeedForwardLayer::backward(const MatrixXf& input){
  const E::MatrixXf& error=next->inputErrorRef();
  //std::cout<<"Next error: "<<error.array().pow(2).mean()<<std::endl;
  const E::MatrixXf& in=(previous==nullptr)?input:previous->outputRef();
  //std::cout<<"Input error: "<<in.array().pow(2).mean()<<std::endl;
  if(previous!=nullptr){
    MatFunction func=previous->getFDot();
    input_error=(weights.transpose()*error).cwiseProduct(func(in));
    //std::cout<<"Error div: "<<input_error.array().pow(2).mean()<<std::endl;
  }
  updateWeights(in,error);

}

// For updating
void FeedForwardLayer::updateWeights(const MatrixXf& input,
                                     const MatrixXf& error){
  /*
  std::cout<<"Input: "<<input.rows()<<" "<<input.cols()<<std::endl;
  std::cout<<"Error: "<<error.rows()<<" "<<error.cols()<<std::endl;
  */
  MatrixXf weight_grads=error*input.transpose();
  MatrixXf bias_grads=error.rowwise().sum();
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
