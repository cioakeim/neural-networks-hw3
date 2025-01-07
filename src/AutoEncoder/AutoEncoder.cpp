#include "AutoEncoder/AutoEncoder.hpp"
#include "MLP/FeedForwardLayer.hpp"
#include "AutoEncoder/MSELayer.hpp"
#include "MLP/ActivationFunctions.hpp"
#include <iostream>


void AutoEncoder::convertToMLP(){
  layers.clear();
  int sz=enc_stack.size();
  for(int i=0;i<sz;i++){
    layers.push_back(enc_stack[i]);
  }
  for(int i=sz-1;i>=0;i--){
    layers.push_back(dec_stack[i]);
  }
  // Configure correct interface types
  for(auto& interface: enc_interfaces){
    interface->type=Hidden;
  }
  for(auto& interface: dec_interfaces){
    interface->type=Hidden;
  }
  encoded_product->type=Hidden;
  // Exceptions
  enc_interfaces[0]->type=Input;
  dec_interfaces[0]->type=Output;
}


void AutoEncoder::addInterfaceStack(InterfacePtr new_encoded_interface){
  // If first entry just insert and return
  if(!encoded_product){
    encoded_product=new_encoded_interface;
    std::cout<<"FIRST LAYER"<<std::endl;
    return;
  }
  // Create an exact copy of current encoded interface
  enc_interfaces.push_back(std::make_shared<LayerInterface>(*encoded_product));
  dec_interfaces.push_back(std::make_shared<LayerInterface>(*encoded_product));
  /*
  dec_interfaces.back()->f=linear;
  dec_interfaces.back()->f_dot=linearder;
  */

  // Link the last layers to the new interfaces
  if(enc_interfaces.size()>1){
    std::cout<<"Entering here"<<std::endl;
    enc_stack.back()->setOutputInterface(enc_interfaces.back());
    dec_stack.back()->setInputInterface(dec_interfaces.back());
  }

  // Update current encoding product 
  encoded_product=new_encoded_interface;
  std::cout<<"DONE"<<std::endl;
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
  std::cout<<"Interface size: "<<if_sz<<std::endl;
  LayerConfig config;
  config.properties=properties;
  // For encoding stack
  config.input_interface=enc_interfaces.back();
  config.output_interface=encoded_product;
  std::cout<<"Here good"<<std::endl;
  enc_stack.back()->configure(config);
  // For decoding stack
  config.input_interface=encoded_product;
  config.output_interface=dec_interfaces.back();
  std::cout<<"Here good"<<std::endl;
  dec_stack.back()->configure(config);

  // Update MLP 
  convertToMLP();
  std::cout<<"Converted"<<std::endl;
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
