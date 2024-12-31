#include "MLP/MLP.hpp"
#include "MLP/FeedForwardLayer.hpp"
#include "MLP/SoftMaxLayer.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <fstream> 
#include <filesystem>
#include <string>
#include <iostream>
#include <random>

namespace fs=std::filesystem;


void MLP::addLayer(LayerConfig config){
  // Create
  switch(config.layer_type){
  case FeedForward:
    layers.push_back(std::make_shared<FeedForwardLayer>());
    break;
  case SoftMax:
    layers.push_back(std::make_shared<SoftMaxLayer>());
    break;
  default:
    std::cerr<<"Don't know what type that is.."<<std::endl;
    exit(1);
    break;
  }
  // Chain
  int last=layers.size()-1;
  if(last!=0){
    layers[last]->setPreviousLayer(layers[last-1]); 
    layers[last-1]->setNextLayer(layers[last]); 
  }
  layers[last]->configure(config);
}


void MLP::setStorePath(std::string path){
  this->store_path=path;
  int size=layers.size();
  for(int i=0;i<size;i++){
    layers[i]->setStorePath(store_path+"/layer_"+std::to_string(i));
  }
}


void MLP::forward(const MatrixXf& input){
  layers[0]->forward(input);
  int size=layers.size();
  for(int i=1;i<size;i++){
    layers[i]->forward();
  }
}


void MLP::backward(const MatrixXf& input,
                   const VectorXi& labels){
  int size=layers.size();
  layers[size-1]->backward(input,labels);
  for(int i=size-2;i>=0;i--){
    layers[i]->backward(input);
  }
}


float MLP::runEpoch(){
  float total_loss=0;
  shuffleDatasetInPlace(training_set);
  int training_sz=training_set.vectors.cols();
  int last_idx=layers.size()-1;
  for(int batch_idx=0;batch_idx<training_sz;batch_idx+=batch_size){
    const MatrixXf& input=training_set.vectors.middleCols(batch_idx,batch_size);
    const VectorXi& labels=training_set.labels.segment(batch_idx,batch_size);
    //std::cout<<"Batch start"<<std::endl;
    forward(input);
    //std::cout<<"Backward start"<<std::endl;
    total_loss+=layers[last_idx]->loss(labels);
    backward(input,labels);
    //std::cout<<"Backward end"<<std::endl;
  }  
  return total_loss/training_sz;
}


// Test the epoch result (return the loss function and accuracy)
void MLP::testModel(float& J_test,float& accuracy){
  J_test=0;
  int success_count=0;
  int test_sz=training_set.vectors.cols();
  int last_idx=layers.size()-1;
  for(int batch_idx=0;batch_idx<test_sz;batch_idx+=batch_size){
    const MatrixXf& input=training_set.vectors.middleCols(batch_idx,batch_size);
    const VectorXi& labels=training_set.labels.segment(batch_idx,batch_size);
    forward(input);
    J_test+=layers[last_idx]->loss(labels);
    success_count+=layers[last_idx]->prediction_success(labels);
  }  
  J_test/=test_sz;
  accuracy=static_cast<float>(success_count)/test_sz;
}



// I/O
void MLP::store(){
  std::cout<<"Storing network"<<std::endl;
  // Create directory
  ensure_a_path_exists(store_path);
  for(size_t i=0;i<layers.size();i++){
    layers[i]->store();
  }
}


void MLP::load(){
  ensure_a_path_exists(store_path);
  for(size_t i=0;i<layers.size();i++){
    layers[i]->load();
  }
}


// Config:
void MLP::init(){
  int size=layers.size();
  for(int i=0;i<size;i++){
    layers[i]->init();
  }
}

