#include "MLP/MLP.hpp"
#include "MLP/FeedForwardLayer.hpp"
#include "MLP/SoftMaxLayer.hpp"
#include "AutoEncoder/MSELayer.hpp"
#include "CommonLib/basicFuncs.hpp"
#include <fstream> 
#include <filesystem>
#include <string>
#include <iostream>
#include <random>

namespace fs=std::filesystem;


void MLP::addLayer(LayerProperties properties){
  int last=interfaces.size()-1;
  LayerConfig config;
  config.properties=properties;
  config.output_interface=interfaces[last];
  config.input_interface=interfaces[last-1];

  // Create
  switch(properties.layer_type){
  case FeedForward:
    layers.push_back(std::make_shared<FeedForwardLayer>());
    break;
  case SoftMax:
    layers.push_back(std::make_shared<SoftMaxLayer>());
    break;
  case MSE:
    layers.push_back(std::make_shared<MSELayer>());
    break;
  default:
    std::cerr<<"Don't know what type that is.."<<std::endl;
    exit(1);
    break;
  }
  layers.back()->configure(config);
}


void MLP::addInterface(std::shared_ptr<LayerInterface> interface){
  interfaces.push_back(interface);
  int size=interfaces.size();
  switch(size){
  case 1:
    interfaces.back()->type=Input;
    break;
  case 2:
    interfaces.back()->type=Output;
    break;
  default:
    interfaces.back()->type=Output;
    interfaces[size-2]->type=Hidden;
    break;
  }
}


void MLP::setStorePath(std::string path){
  this->store_path=path;
  int size=layers.size();
  for(int i=0;i<size;i++){
    layers[i]->setStorePath(store_path+"/layer_"+std::to_string(i));
  }
}


void MLP::forward(const PassContext& context){
  int size=layers.size();
  for(int i=0;i<size;i++){
    layers[i]->forward(context);
  }
}


void MLP::backward(const PassContext& context){
  int size=layers.size();
  for(int i=size-1;i>=0;i--){
    layers[i]->backward(context);
  }
}


float MLP::runEpoch(){
  float total_loss=0;
  shuffleDatasetInPlace(training_set);
  int training_sz=training_set.vectors.cols();
  int last_idx=layers.size()-1;
  for(int batch_idx=0;batch_idx<training_sz;batch_idx+=batch_size){
    const PassContext context{
      training_set.vectors.middleCols(batch_idx,batch_size),
      training_set.labels.segment(batch_idx,batch_size)
    };
    forward(context);
    total_loss+=layers[last_idx]->loss(context);
    backward(context);
  }  
  for(auto& layer: layers){
    layer->printStateInfo();
  }
  
  return total_loss/training_sz;
}


// Test the epoch result (return the loss function and accuracy)
void MLP::testModel(const SampleMatrix& set,
                    float& J_test,float& accuracy){
  J_test=0;
  int success_count=0;
  int test_sz=set.vectors.cols();
  int last_idx=layers.size()-1;
  for(int batch_idx=0;batch_idx<test_sz;batch_idx+=batch_size){
    const PassContext context{
      set.vectors.middleCols(batch_idx,batch_size),
      set.labels.segment(batch_idx,batch_size)
    };
    forward(context);
    J_test+=layers[last_idx]->loss(context);
    success_count+=layers[last_idx]->prediction_success(context);
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

