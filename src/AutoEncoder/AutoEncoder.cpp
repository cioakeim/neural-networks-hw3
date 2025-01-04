#include "AutoEncoder/AutoEncoder.hpp"
#include "MLP/FeedForwardLayer.hpp"
#include "AutoEncoder/MSELayer.hpp"
#include <iostream>


void AutoEncoder::convertToMLP(){
  layers.clear();
  for(auto it= enc_stack.begin();it!=enc_stack.end();it++){
    layers.push_back(*it);
  }
  for(auto it= dec_stack.rbegin();it!=enc_stack.rend();it++){
    layers.push_back(*it);
  }
}


void AutoEncoder::addInterfaceStack(InterfacePtr new_encoded_interface){
  // If first entry just insert and return
  if(enc_interfaces.size()==0){
    encoded_product=new_encoded_interface;
    return;
  }
  // Create an exact copy of current encoded interface
  enc_interfaces.push_back(encoded_product);
  dec_interfaces.push_back(std::make_shared<LayerInterface>(*encoded_product));

  // Link the last layers to the new interfaces
  enc_stack.back()->setOutputInterface(enc_interfaces.back());
  dec_stack.back()->setOutputInterface(dec_interfaces.back());

  // Update current encoding product 
  encoded_product=new_encoded_interface;
}


void AutoEncoder::addLayerStack(LayerProperties properties){
  // Adding a layer means locking the rest 
  if(enc_stack.size()>0){
    enc_stack.back()->lock();
    dec_stack.back()->lock();
  }
  // Push on both stacks
  switch(properties.layer_type){
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
  // Configure both layers
  int if_sz=enc_interfaces.size();
  LayerConfig config;
  config.properties=properties;
  // For encoding stack
  config.input_interface=enc_interfaces[if_sz-1];
  config.output_interface=encoded_product;
  enc_stack.back()->configure(config);
  // For decoding stack
  config.input_interface=encoded_product;
  config.input_interface=dec_interfaces[if_sz-1];
  dec_stack.back()->configure(config);

  // Update MLP 
  convertToMLP();
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
