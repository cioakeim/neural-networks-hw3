#include "AutoEncoder/AutoEncoder.hpp"
#include "MLP/FeedForwardLayer.hpp"
#include "AutoEncoder/MSELayer.hpp"
#include <iostream>
#include <ranges>


/*
void AutoEncoder::convertToMLP(){
  layers.clear();
  for(auto it= enc_stack.begin();it!=enc_stack.end();it++){
    layers.push_back(*it);
  }
  for(auto it= dec_stack.rbegin();it!=enc_stack.rend();it++){
    layers.push_back(*it);
  }
}


void AutoEncoder::addStackLayer(LayerConfig config){
  LayerConfig enc_config,dec_config;
  enc_config=dec_config=config;
  // Push on both stacks
  switch(config.layer_type){
  case FeedForward:
    enc_stack.push_back(std::make_shared<FeedForwardLayer>());
    dec_stack.push_back(std::make_shared<FeedForwardLayer>());
    break;
  case MSE:
    enc_stack.push_back(std::make_shared<FeedForwardLayer>());
    dec_stack.push_back(std::make_shared<MSELayer>());
    break;
  default:
    std::cerr<<"This layer type isn't supported.. Abort."<<std::endl;
    break;
  }
  
}


MatrixXf AutoEncoder::encode(MatrixXf& set){
  const int enc_size=enc_stack.size();
  const PassContext context{
    set,VectorXi::Zero(0)
  };
  for(int i=0;i<enc_size;i++){
    enc_stack[i]->forward(context);
  }
  return enc_stack[enc_size-1]->getOutputInterface()->forward_signal;
}
*/
